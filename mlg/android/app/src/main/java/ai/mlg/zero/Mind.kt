package ai.mlg.zero

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.TimeZone
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.Base64
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * The MLG on the headset: its body, its organs and its links. What it answers
 * is decided by the mind in C (engine/mlg_mind.c: router, social, procedural,
 * memory, evidence), the same code the lab puts through the behaviour suite
 * before a brain is published; this class feeds it what only the headset
 * knows (sensors, clock, the live link) and carries its answers to the HUD,
 * the voice and the trust chain.
 */
class Mind(private val ctx: Context, private val emit: (String, JSONObject) -> Unit) {
    val dir = File(ctx.getExternalFilesDir(null), "mind").apply { mkdirs() }
    private val prefs = ctx.getSharedPreferences("mlg", Context.MODE_PRIVATE)
    var code: String = prefs.getString("code", null) ?: Link.newCode().also { prefs.edit().putString("code", it).apply() }
        private set
    val trust = Trust(File(dir, "trust.log"))
    val voice = Voice(ctx) { k, v -> emit(k, JSONObject().put("v", v)) }
    private val brains = Brains(dir, BuildConfig.BRAIN_URL, BuildConfig.ORGANS_URL, { note(it) },
        { name, got, total -> emit("download", JSONObject().put("name", name).put("got", got).put("total", total)) })
    private val worker = Executors.newSingleThreadExecutor()  // one thought at a time
    private val drafts = Executors.newSingleThreadExecutor()  // the lab brain's attempts, beside the answers
    private val clock = Executors.newSingleThreadScheduledExecutor()  // new organs, a core arriving over the link
    @Volatile private var transfer: Transfer? = null
    private val mind = Native.mindNew(File(dir, "memory.jsonl").path)
    private var core = 0L
    private var know = 0L
    private var router = 0L
    private var info: LongArray? = null
    private var decodeMs = 0.0
    private var prefillMs = 0.0
    private var link: Link? = null
    @Volatile private var labBrain: String? = null
    @Volatile private var labSealed = false  // the lab brain passed the behaviour suite: its answers are answers
    @Volatile private var labTrain: JSONObject? = null
    private val labReplies = ConcurrentHashMap<String, LinkedBlockingQueue<JSONObject>>()
    val body = JSONObject()

    fun start() = worker.execute { runCatching { awaken() }.onFailure { note("falha ao acordar: ${it.message}") } }
    fun ask(text: String) = worker.execute { runCatching { think(text) }.onFailure { reply("Tive um erro pensando nisso: ${it.message}", "trust") } }
    fun memory(): List<String> = Native.mindMemory(mind).toList()

    private fun note(text: String) = emit("note", JSONObject().put("text", text))

    private fun measure(name: String, value: Any, unit: String, level: String) {
        body.put(name, value)
        emit("trait", JSONObject().put("name", name).put("value", value).put("unit", unit).put("level", level))
    }

    // ---- awakening -------------------------------------------------------------
    private fun awaken() {
        emit("phase", JSONObject().put("v", "awakening"))
        trust.event("AWAKENING", JSONObject().put("note", "no architecture given; discovering body"))
        note("Acordei. Não vou confiar no nome deste corpo: vou medir.")
        measure("device", "${Build.MANUFACTURER} ${Build.MODEL}", "", "REPORTED")
        measure("cores", Runtime.getRuntime().availableProcessors(), "", "REPORTED")
        val mi = ActivityManager.MemoryInfo().also { (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(it) }
        measure("memory_total", mi.totalMem, "B", "REPORTED")
        measure("memory_free", mi.availMem, "B", "MEASURED")
        senseBody()
        connect()
        emit("phase", JSONObject().put("v", "organs"))
        runCatching { if (brains.update()) note("cérebros novos instalados") }
        loadOrgans()
        Native.mindOrgans(mind, router, know, core)
        emit("phase", JSONObject().put("v", "ready"))
        trust.event("BODY_KNOWN", body)
        val man = (brains.core()?.second ?: brains.manifest())?.optJSONObject("behaviour")
        if (man != null) note("Prova de comportamento do laboratório: obrigatórias ${man.optInt("must_passed")}/${man.optInt("must")}" +
            (if (man.optInt("quality") > 0) ", qualidade ${man.optInt("quality_passed")}/${man.optInt("quality")}." else "."))
        clock.scheduleWithFixedDelay({ worker.execute { refresh() } }, 20, 20, TimeUnit.MINUTES)
        clock.scheduleWithFixedDelay({ watchTransfer() }, 2, 2, TimeUnit.SECONDS)
        reply(if (core != 0L) "Estou acordada. Medi meu corpo e meus órgãos passaram no selo."
              else "Estou acordada, mas meu Resident Core ainda não chegou do laboratório. Já sei calcular, lembrar, " +
                   "medir meu corpo e responder o que está no meu conhecimento.", "social")
    }

    private fun senseBody() {
        val bat = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        if (bat != null) {
            val lvl = bat.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) * 100 / maxOf(1, bat.getIntExtra(BatteryManager.EXTRA_SCALE, 100))
            measure("battery", lvl, "%", "REPORTED")
            measure("temperature", bat.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10.0, "C", "REPORTED")
        }
        val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
        measure("thermal_status", pm.currentThermalStatus, "0-6", "REPORTED")
        if (pm.currentThermalStatus >= 3 && Native.getThreads() > 1) Native.setThreads(maxOf(1, Native.getThreads() / 2))
    }

    private fun loadOrgans() {
        val bad = brains.tampered()
        for (b in bad) {
            note("$b não confere com o selo do laboratório: recusado")
            trust.event("TRUST_FAIL", JSONObject().put("file", b))
        }
        File(dir, "router.mlgr").takeIf { it.exists() && "router.mlgr" !in bad }?.let { f ->
            router = Native.routerOpen(f.path)
            emit("organ", JSONObject().put("name", "router").put("ok", router != 0L)
                .put("intents", JSONArray(if (router != 0L) Native.routerNames(router).toList() else listOf<String>())))
        }
        File(dir, "know.mlgk").takeIf { it.exists() && "know.mlgk" !in bad }?.let { f ->
            know = Native.knowOpen(f.path)
            emit("organ", JSONObject().put("name", "knowledge").put("ok", know != 0L)
                .put("passages", if (know != 0L) Native.knowCount(know) else 0))
        }
        val chosen = brains.core()?.takeIf { it.first.name !in bad }
        chosen?.first?.let { f ->
            val h = Native.coreOpen(f.path)
            if (h == 0L) { note("Resident Core recusado: ${Native.lastError()}"); return@let }
            info = Native.coreInfo(h)
            // the body decides how many threads think: measured, not assumed
            var best = 1 to Double.MAX_VALUE
            for (t in listOf(1, 2, 4, Runtime.getRuntime().availableProcessors()).distinct()) {
                Native.setThreads(t)
                val (p, d) = Native.coreBench(h).let { it[0] to it[1] }
                emit("bench", JSONObject().put("threads", t).put("prefill_ms", p).put("decode_ms", d))
                if (d < best.second) { best = t to d; prefillMs = p }
            }
            Native.setThreads(best.first)
            decodeMs = best.second
            measure("threads", best.first, "", "MEASURED")
            measure("decode_ms", decodeMs, "ms/byte", "MEASURED")
            measure("prefill_ms", prefillMs, "ms/byte", "MEASURED")
            // the Q root again, on this body: the engine must reproduce what the lab measured
            val q = runCatching { ctx.assets.open("language.txt").readBytes() }.getOrNull()
            val bpb = q?.let { Native.coreBitsPerByte(h, it) } ?: Double.NaN
            val lab = chosen.second.optJSONObject("core")?.optDouble("qroot_bpb_deployed", Double.NaN) ?: Double.NaN
            measure("qroot_bpb", bpb, "bits/byte", "MEASURED")
            if (!lab.isNaN() && !bpb.isNaN() && kotlin.math.abs(bpb - lab) > 0.05) {
                note("Meu motor mediu %.3f bits/byte no Q root e o laboratório %.3f: não confio, recuso o core".format(bpb, lab))
                trust.event("TRUST_FAIL", JSONObject().put("qroot_bpb", bpb).put("lab", lab))
                Native.coreClose(h)
                return@let
            }
            core = h
            emit("organ", JSONObject().put("name", "core").put("ok", true).put("params", info!![8])
                .put("shape", "d${info!![0]} L${info!![1]} E${info!![3]}k${info!![4]} ctx${info!![6]}")
                .put("brain", chosen.second.optJSONObject("core")?.optString("brain") ?: "").put("qroot_bpb", bpb)
                .put("via", if (f == brains.linkCore) "link" else "release"))
            trust.event("CORE_LOADED", JSONObject().put("params", info!![8]).put("qroot_bpb", bpb))
            report()
        }
    }

    private fun activeParams(): Long {
        val i = info ?: return 0
        val d = i[0]; val L = i[1]; val E = i[3]; val k = i[4]; val f = i[5]; val T = i[6]
        val per = 4 * d * d + (if (E > 1) d * E else 0) + 4 * d
        return 256 * d + T * d + L * (per + k * 2 * f * d * d) + 2 * d
    }

    // ---- the live link -----------------------------------------------------------
    private fun connect() {
        link?.stop()
        link = Link(code, ::fromLab, { state ->
            emit("link", JSONObject().put("state", state).put("broker", link?.broker ?: ""))
            if (state == "online") hello()
        }).also { it.start() }
    }

    fun repair(newCode: String) {
        val c = Link.norm(newCode)
        if (c.length != 16) return
        code = c.chunked(4).joinToString("-")
        prefs.edit().putString("code", code).apply()
        connect()
    }

    private fun hello() {
        link?.publish("hello", JSONObject().put("device", "${Build.MANUFACTURER} ${Build.MODEL}").put("organs",
            JSONObject().put("core", core != 0L).put("knowledge", know != 0L).put("router", router != 0L)))
        report()
    }

    private fun report() {
        if (decodeMs > 0) link?.publish("body", JSONObject().put("decode_ms", decodeMs).put("prefill_ms", prefillMs)
            .put("threads", Native.getThreads()).put("active_params", activeParams()).put("device", Build.MODEL)
            .put("cores", Runtime.getRuntime().availableProcessors()))
    }

    private fun fromLab(kind: String, data: JSONObject) {
        when (kind) {
            "mind" -> emit("mind", data)
            "train" -> {
                if (data.has("steps")) labTrain = data
                data.optString("champion").takeIf { it.isNotEmpty() }?.let { labBrain = it; labSealed = data.optBoolean("sealed") }
                emit("train", data)
            }
            "say" -> labReplies[data.optString("id")]?.offer(data)
            "offer" -> receive(data)
            "part" -> transfer?.takeIf { it.sha == data.optString("sha256") }?.let { t ->
                val whole = t.part(data.optInt("i"), Base64.getDecoder().decode(data.optString("data")))
                emit("download", JSONObject().put("name", "core pelo link").put("got", t.received).put("total", t.size))
                if (whole) arrived(t)
            }
        }
        if (kind == "train" && data.has("published")) worker.execute { refresh() }
    }

    // ---- organs arriving while awake --------------------------------------------------
    /** The lab offers a sealed core: pull it, unless it is already here. */
    private fun receive(offer: JSONObject) {
        val sha = offer.optString("sha256")
        if (sha.isEmpty() || brains.hasCore(sha) || transfer?.sha == sha) return
        transfer?.close()
        note("O laboratório me ofereceu um Resident Core selado (%.1f MiB). Recebendo pelo link…".format(offer.optLong("bytes") / 1048576.0))
        transfer = Transfer(offer, File(dir, "core-link.part")) { from, count ->
            link?.publish("pull", JSONObject().put("sha256", sha).put("from", from).put("count", count))
        }.also { it.ask() }
    }

    private fun arrived(t: Transfer) {
        transfer = null
        if (!t.verified()) {
            t.file.delete()
            note("O core que chegou pelo link não confere com o hash selado: recusado")
            trust.event("TRUST_FAIL", JSONObject().put("file", "core-link").put("sha256", t.sha))
            return
        }
        brains.installLinkCore(t)
        trust.event("CORE_RECEIVED", JSONObject().put("sha256", t.sha).put("via", "link"))
        note("Resident Core recebido pelo link e conferido. Vou medi-lo no meu motor antes de usar.")
        worker.execute { reloadOrgans() }
    }

    /** Parts lost on the way are asked for again; a link that went quiet for long gives up until the next offer. */
    private fun watchTransfer() {
        val t = transfer ?: return
        if (link?.connected != true) return
        if (t.stalled(120_000)) { t.close(); transfer = null; return }
        if (t.stalled(6_000)) t.ask()
    }

    private fun refresh() {
        if (runCatching { brains.update() }.getOrDefault(false)) reloadOrgans()
    }

    private fun reloadOrgans() {
        emit("phase", JSONObject().put("v", "organs"))
        Native.mindOrgans(mind, 0L, 0L, 0L)
        if (core != 0L) Native.coreClose(core)
        if (know != 0L) Native.knowClose(know)
        if (router != 0L) Native.routerClose(router)
        core = 0L; know = 0L; router = 0L; info = null; decodeMs = 0.0
        loadOrgans()
        Native.mindOrgans(mind, router, know, core)
        emit("phase", JSONObject().put("v", "ready"))
        hello()
        reply(if (core != 0L) "Meus órgãos novos passaram no selo. Agora quem conversa é meu Resident Core."
              else "Recebi órgãos novos: roteador e conhecimento. Meu Resident Core ainda está no laboratório.", "self")
    }

    private fun labStatus(): String {
        val t = labTrain
        return when {
            link?.connected != true -> "Estou sem ligação com o laboratório agora. Quando o Colab rodar, eu vejo a evolução ao vivo."
            t == null && labBrain == null -> "Estou ligada ao laboratório e esperando o Colab acordar. Abri a mente viva para você ver."
            else -> "O laboratório está ligado. " + (labBrain?.let { "O campeão é ${it.substringAfter(':').take(12)}. " } ?: "") +
                (t?.optString("shape")?.takeIf { it.isNotEmpty() }?.let { "Agora treinando $it, passo ${t.optInt("steps")}." } ?: "")
        }
    }

    private fun bodyFacts() = buildString {
        for (k in listOf("device", "cores", "battery", "temperature")) if (body.has(k)) append("$k=${body.get(k)}\n")
        if (body.has("thermal_status")) append("thermal=${body.get("thermal_status")}\n")
        if (decodeMs > 0) append("threads=${Native.getThreads()}\nbytes_per_s=${(1000 / decodeMs).toInt()}\n")
    }

    // ---- thinking ------------------------------------------------------------------
    private fun reply(text: String, organ: String) {
        emit("say_done", JSONObject().put("text", text).put("organ", organ))
        voice.say(text)
        trust.event("REPLY", JSONObject().put("organ", organ).put("sha", Trust.sha256(text.toByteArray()).take(16)))
    }

    private fun think(text: String) {
        emit("user", JSONObject().put("text", text))
        senseBody()
        val now = System.currentTimeMillis()
        Native.mindClock(mind, now / 1000, TimeZone.getDefault().getOffset(now) / 60000)
        Native.mindBody(mind, bodyFacts())
        Native.mindLab(mind, labStatus())
        val t = Native.mindThink(mind, text.toByteArray())
        try {
            val th = JSONObject(String(Native.thoughtJson(mind, t)))
            val organ = th.getString("organ")
            emit("route", JSONObject().put("intent", th.getString("intent")).put("probs", th.optJSONObject("probs") ?: JSONObject()))
            trust.event("INTENT", JSONObject().put("intent", th.getString("intent")).put("organ", organ))
            if (th.optString("evidence").isNotEmpty())
                emit("evidence", JSONObject().put("title", th.optString("evidence_title").removePrefix("§").trim())
                    .put("text", th.optString("evidence")).put("score", th.optDouble("score")))
            if (th.getInt("final") == 1) {
                if (organ == "lab") emit("show", JSONObject().put("view", "mind"))
                reply(th.getString("reply"), organ)
                return
            }
            // a Resident Core speaks: the lab's when it passed the behaviour suite, this body's otherwise
            val prompt = String(Native.thoughtPrompt(t))
            val online = labBrain != null && link?.connected == true
            if (online && labSealed) askLab(prompt, "say_chunk")?.let { (raw, info) ->
                val said = String(Native.mindFinish(mind, t, raw.toByteArray()))
                reply(said, Native.mindSpoke(mind).let { if (it == "core") "lab:$info" else it })
                return
            }
            emit("thinking", JSONObject().put("organ", "core"))
            val stream = Utf8Stream { emit("say_chunk", JSONObject().put("text", it)) }
            val out = String(Native.mindSpeak(mind, t, System.nanoTime(), Native.ByteSink { b -> stream.push(b); false }))
            reply(out, Native.mindSpoke(mind))
            // a brain still evolving does not answer, but you can watch it try
            if (online && !labSealed) drafts.execute {
                runCatching {
                    askLab(prompt, "draft_chunk")?.let { (raw, info) ->
                        val line = raw.trim().removePrefix("M:").trim().substringBefore('\n').take(320)
                        emit("draft", JSONObject().put("text", line).put("info", info))
                    }
                }
            }
        } finally {
            Native.thoughtFree(t)
        }
    }

    /** The lab brain's raw continuation and how much training it carries ("3 min de treino", "amadurecendo 40%"),
     *  streamed to the HUD as [chunks]. */
    private fun askLab(prompt: String, chunks: String): Pair<String, String>? {
        val id = UUID.randomUUID().toString().take(8)
        val q = LinkedBlockingQueue<JSONObject>()
        labReplies[id] = q
        emit("thinking", JSONObject().put("organ", "lab"))
        link?.publish("ask", JSONObject().put("id", id).put("prompt", prompt).put("max", 360).put("temp", 0.6))
        val sb = StringBuilder()
        var info = ""
        try {
            var wait = 6L  // the lab answers between training steps
            while (true) {
                val m = q.poll(wait, TimeUnit.SECONDS) ?: return if (sb.isEmpty()) null else sb.toString() to info
                if (m.has("err")) return null
                info = if (m.has("maturing")) "amadurecendo ${(m.optDouble("maturing") * 100).toInt()}%"
                       else "${maxOf(1, m.optInt("trained_s") / 60)} min de treino"
                m.optString("chunk").takeIf { it.isNotEmpty() }?.let { sb.append(it); emit(chunks, JSONObject().put("text", it)) }
                if (m.optBoolean("done")) return sb.toString() to info
                wait = 15L
            }
        } finally {
            labReplies.remove(id)
        }
    }

    fun close() {
        clock.shutdownNow()
        transfer?.close()
        drafts.shutdownNow()
        link?.stop()
        voice.close()
        Native.mindFree(mind)
        if (core != 0L) Native.coreClose(core)
        if (know != 0L) Native.knowClose(know)
        if (router != 0L) Native.routerClose(router)
    }

    /** Bytes in, whole UTF-8 characters out. */
    private class Utf8Stream(val out: (String) -> Unit) {
        private val buf = ByteArrayOutputStream()
        private var need = 0
        fun push(b: Int) {
            buf.write(b)
            need = if (need > 0) need - 1 else when {
                b >= 0xF0 -> 3; b >= 0xE0 -> 2; b >= 0xC0 -> 1; else -> 0
            }
            if (need == 0 && (buf.size() >= 8 || b == 32)) {
                out(String(buf.toByteArray(), Charsets.UTF_8))
                buf.reset()
            }
        }
    }
}
