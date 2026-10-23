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
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * The MLG on the headset. It wakes up knowing nothing about this body,
 * measures it, loads the organs the lab grew (checking every seal), and then
 * answers each intent with the organ the router picks:
 *
 *   math, time        procedural expert (exact)
 *   remember, recall  persistent memory
 *   body              its own sensors
 *   lab               the live link to the Colab lab
 *   knowledge, chat   retrieval + Resident Core; the lab's bigger brain when the link is up
 *
 * Everything it does goes to the HUD as events, and into the trust chain.
 */
class Mind(private val ctx: Context, private val emit: (String, JSONObject) -> Unit) {
    val dir = File(ctx.getExternalFilesDir(null), "mind").apply { mkdirs() }
    private val prefs = ctx.getSharedPreferences("mlg", Context.MODE_PRIVATE)
    var code: String = prefs.getString("code", null) ?: Link.newCode().also { prefs.edit().putString("code", it).apply() }
        private set
    val memory = Memory(File(dir, "memory.json"))
    val trust = Trust(File(dir, "trust.log"))
    val voice = Voice(ctx) { k, v -> emit(k, JSONObject().put("v", v)) }
    private val brains = Brains(dir, BuildConfig.BRAIN_URL, { note(it) },
        { name, got, total -> emit("download", JSONObject().put("name", name).put("got", got).put("total", total)) })
    private val worker = Executors.newSingleThreadExecutor()  // one thought at a time
    private var core = 0L
    private var know = 0L
    private var router = 0L
    private var intents = arrayOf<String>()
    private var info: LongArray? = null
    private var decodeMs = 0.0
    private var prefillMs = 0.0
    private var link: Link? = null
    @Volatile private var labBrain: String? = null
    @Volatile private var labTrain: JSONObject? = null
    private val labReplies = ConcurrentHashMap<String, LinkedBlockingQueue<JSONObject>>()
    val body = JSONObject()

    fun start() = worker.execute { runCatching { awaken() }.onFailure { note("falha ao acordar: ${it.message}") } }
    fun ask(text: String) = worker.execute { runCatching { think(text) }.onFailure { reply("Tive um erro pensando nisso: ${it.message}", "trust") } }

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

        link = Link(code, ::fromLab, { state ->
            emit("link", JSONObject().put("state", state).put("broker", link?.broker ?: ""))
            if (state == "online") hello()
        }).also { it.start() }

        emit("phase", JSONObject().put("v", "organs"))
        runCatching { if (brains.update()) note("cérebros novos instalados") }
        loadOrgans()
        emit("phase", JSONObject().put("v", "ready"))
        trust.event("BODY_KNOWN", body)
        val name = memory.recall("meu nome", 1).firstOrNull { it.key == "nome" }?.text
        reply(if (core != 0L) "Estou acordada${if (name != null) ", ${name.substringAfterLast(' ')}" else ""}. Medi meu corpo e meus órgãos passaram no selo."
              else "Estou acordada, mas meu Resident Core ainda não chegou do laboratório. Já sei calcular, lembrar e medir meu corpo.", "chat")
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
    }

    private fun loadOrgans() {
        val bad = brains.tampered()
        for (b in bad) {
            note("$b não confere com o selo do laboratório: recusado")
            trust.event("TRUST_FAIL", JSONObject().put("file", b))
        }
        val man = brains.manifest()
        File(dir, "router.mlgr").takeIf { it.exists() && "router.mlgr" !in bad }?.let { f ->
            router = Native.routerOpen(f.path)
            if (router != 0L) intents = Native.routerNames(router)
            emit("organ", JSONObject().put("name", "router").put("ok", router != 0L).put("intents", JSONArray(intents.toList())))
        }
        File(dir, "know.mlgk").takeIf { it.exists() && "know.mlgk" !in bad }?.let { f ->
            know = Native.knowOpen(f.path)
            emit("organ", JSONObject().put("name", "knowledge").put("ok", know != 0L)
                .put("passages", if (know != 0L) Native.knowCount(know) else 0))
        }
        File(dir, "core.mlgb").takeIf { it.exists() && "core.mlgb" !in bad }?.let { f ->
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
            val lab = man?.optJSONObject("core")?.optDouble("qroot_bpb_deployed", Double.NaN) ?: Double.NaN
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
                .put("brain", man?.optJSONObject("core")?.optString("brain") ?: "").put("qroot_bpb", bpb))
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
                labTrain = data
                data.optString("champion").takeIf { it.isNotEmpty() }?.let { labBrain = it }
                emit("train", data)
            }
            "say" -> labReplies[data.optString("id")]?.offer(data)
        }
    }

    // ---- thinking ------------------------------------------------------------------
    private fun route(text: String): Pair<String, JSONObject> {
        val probs = JSONObject()
        if (router != 0L) {
            val p = Native.routerRoute(router, text.toByteArray())
            intents.forEachIndexed { i, n -> probs.put(n, p[i].toDouble()) }
            return intents[p.indices.maxByOrNull { p[it] } ?: 0] to probs
        }
        val t = text.lowercase()  // no router yet: the obvious cases only
        val i = when {
            Procedural.math(t) != null -> "math"
            Regex("que horas|que dia|data de hoje").containsMatchIn(t) -> "time"
            Regex("^(lembr|anota|guarda|meu nome|eu me chamo)").containsMatchIn(t) -> "remember"
            Regex("(qual|como) .*(meu nome|me chamo)|o que eu (te )?disse").containsMatchIn(t) -> "recall"
            Regex("bateria|temperatura|seu corpo").containsMatchIn(t) -> "body"
            Regex("evolu|c[eé]rebro|laborat|colab").containsMatchIn(t) -> "lab"
            else -> "knowledge"
        }
        return i to probs.put(i, 1.0)
    }

    private fun reply(text: String, organ: String) {
        emit("say_done", JSONObject().put("text", text).put("organ", organ))
        voice.say(text)
        trust.event("REPLY", JSONObject().put("organ", organ).put("sha", Trust.sha256(text.toByteArray()).take(16)))
    }

    private fun think(text: String) {
        emit("user", JSONObject().put("text", text))
        val (intent, probs) = route(text)
        emit("route", JSONObject().put("intent", intent).put("probs", probs))
        trust.event("INTENT", JSONObject().put("intent", intent))
        when (intent) {
            "math" -> Procedural.math(text)?.let { reply(it, "procedural") } ?: speak(text, null)
            "time" -> reply(Procedural.time(text), "procedural")
            "remember" -> reply("Guardei na minha memória: \"${memory.remember(text)}\".", "memory")
            "recall" -> memory.recall(text).let { f ->
                reply(if (f.isEmpty()) "Ainda não tenho isso guardado. Me diga e eu lembro."
                      else f.joinToString(" ") { "Você me disse: \"${it.text}\"." }, "memory")
            }
            "body" -> {
                senseBody()
                reply(buildString {
                    append("Meu corpo é um ${body.optString("device")} com ${body.optInt("cores")} núcleos. ")
                    append("Bateria em ${body.optInt("battery")}%, ${body.optDouble("temperature")} °C. ")
                    if (decodeMs > 0) append("Penso a %.0f bytes por segundo com %d threads, medido agora.".format(1000 / decodeMs, Native.getThreads()))
                    if (body.optInt("thermal_status") >= 3) append(" Estou quente: vou pensar com menos threads.")
                }, "body")
                if (body.optInt("thermal_status") >= 3) Native.setThreads(maxOf(1, Native.getThreads() / 2))
            }
            "lab" -> {
                emit("show", JSONObject().put("view", "mind"))
                val t = labTrain
                reply(when {
                    link?.connected != true -> "Estou sem ligação com o laboratório agora. Quando o Colab rodar, eu vejo a evolução ao vivo."
                    t == null -> "Estou ligada ao laboratório e esperando o Colab acordar. Abri a mente viva para você ver."
                    else -> "O laboratório está ligado. " + (labBrain?.let { "O campeão é ${it.substringAfter(':').take(12)}. " } ?: "") +
                        (t.optString("shape").takeIf { it.isNotEmpty() }?.let { "Agora treinando $it, passo ${t.optInt("steps")}." } ?: "")
                }, "lab")
            }
            else -> speak(text, if (intent == "knowledge" || intent == "chat") intent else "knowledge")
        }
    }

    /** Retrieval + Resident Core (the lab's when it is up, this body's otherwise). */
    private fun speak(text: String, intent: String?) {
        val ctxParts = mutableListOf<String>()
        if (know != 0L && intent != "chat") {
            val r = Native.knowSearch(know, text.toByteArray(), 2)
            for (i in 0 until r.size / 2) {
                if (r[2 * i + 1] < 4f) continue  // too weak to be evidence
                val passage = String(Native.knowText(know, r[2 * i].toInt()))
                ctxParts += passage.take(420)
                emit("evidence", JSONObject().put("title", String(Native.knowTitle(know, r[2 * i].toInt())))
                    .put("text", passage.take(300)).put("score", r[2 * i + 1].toDouble()))
                if (ctxParts.size == 1 && r.size > 2 && r[3] < r[1] * 0.6f) break
            }
        }
        memory.recall(text, 2).forEach { ctxParts += "(memória) ${it.text}" }
        val prompt = (if (ctxParts.isNotEmpty()) "C: ${ctxParts.joinToString(" ")}\n" else "") + "U: ${text.trim()}\nM:"
        labBrain?.takeIf { link?.connected == true }?.let { brain ->
            askLab(prompt)?.let { reply(it.trim(), "lab:$brain"); return }
        }
        if (core != 0L) {
            emit("thinking", JSONObject().put("organ", "core"))
            val stream = Utf8Stream { emit("say_chunk", JSONObject().put("text", it)) }
            val out = Native.coreGenerate(core, prompt.toByteArray(), 360, 0.6f, 0.9f, System.nanoTime(),
                Native.ByteSink { b -> stream.push(b); false })
            reply(String(out).trim().ifEmpty { "..." }, "core")
        } else if (ctxParts.isNotEmpty() && !ctxParts[0].startsWith("(memória)")) {
            reply("Pelo que eu sei: " + ctxParts[0].split(Regex("(?<=[.!?])\\s+")).take(2).joinToString(" "), "knowledge")
        } else {
            reply("Meu Resident Core ainda não chegou do laboratório, e não achei isso no meu conhecimento.", "trust")
        }
    }

    private fun askLab(prompt: String): String? {
        val id = UUID.randomUUID().toString().take(8)
        val q = LinkedBlockingQueue<JSONObject>()
        labReplies[id] = q
        emit("thinking", JSONObject().put("organ", "lab"))
        link?.publish("ask", JSONObject().put("id", id).put("prompt", prompt).put("max", 360).put("temp", 0.6))
        val sb = StringBuilder()
        try {
            var wait = 6L  // the lab answers between training steps
            while (true) {
                val m = q.poll(wait, TimeUnit.SECONDS) ?: return if (sb.isEmpty()) null else sb.toString()
                if (m.has("err")) return null
                m.optString("chunk").takeIf { it.isNotEmpty() }?.let { sb.append(it); emit("say_chunk", JSONObject().put("text", it)) }
                if (m.optBoolean("done")) return sb.toString()
                wait = 15L
            }
        } finally {
            labReplies.remove(id)
        }
    }

    fun repair(newCode: String) {
        val c = Link.norm(newCode)
        if (c.length != 16) return
        code = c.chunked(4).joinToString("-")
        prefs.edit().putString("code", code).apply()
        link?.stop()
        link = Link(code, ::fromLab, { state ->
            emit("link", JSONObject().put("state", state).put("broker", link?.broker ?: ""))
            if (state == "online") hello()
        }).also { it.start() }
    }

    fun close() {
        link?.stop()
        voice.close()
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
            if (need == 0 && buf.size() >= 8 || b == 32 && need == 0) {
                out(String(buf.toByteArray(), Charsets.UTF_8))
                buf.reset()
            }
        }
    }
}
