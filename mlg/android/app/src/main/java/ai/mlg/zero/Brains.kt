package ai.mlg.zero

import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * The organs the lab grows arrive here: core.mlgb (Resident Core), know.mlgk
 * (retrieval), router.mlgr (router), described by manifest.json with their
 * SHA-256 and what the lab measured. The notebook publishes them to a GitHub
 * release; they can also be copied by hand into Android/data/ai.mlg.zero/files/mind.
 * A file is only swapped in after its hash matches the manifest.
 */
class Brains(val dir: File, private val baseUrl: String, private val log: (String) -> Unit,
             private val progress: (String, Long, Long) -> Unit) {
    val names = listOf("core.mlgb", "know.mlgk", "router.mlgr")

    fun manifest(): JSONObject? = File(dir, "manifest.json").takeIf { it.exists() }
        ?.let { runCatching { JSONObject(it.readText()) }.getOrNull() }

    private fun get(url: String): HttpURLConnection {
        var u = URL(url)
        repeat(6) {  // GitHub release assets redirect to their storage host
            val c = u.openConnection() as HttpURLConnection
            c.connectTimeout = 10_000
            c.readTimeout = 30_000
            c.instanceFollowRedirects = false
            val code = c.responseCode
            if (code in 300..399) { u = URL(u, c.getHeaderField("Location")); c.disconnect() } else return c
        }
        error("too many redirects")
    }

    /** Fetches a newer manifest and every organ whose hash changed. Returns true if anything changed. */
    fun update(): Boolean {
        val remote = runCatching {
            get("$baseUrl/manifest.json").let { c ->
                if (c.responseCode != 200) null else JSONObject(c.inputStream.bufferedReader().readText())
            }
        }.getOrNull() ?: return false
        val local = manifest()
        if (local != null && local.optString("version") == remote.optString("version")) return false
        log("o laboratório publicou cérebros novos (${remote.optString("version")})")
        val files = remote.getJSONObject("files")
        for (name in names) {
            val want = files.optJSONObject(name) ?: continue
            val target = File(dir, name)
            if (target.exists() && local?.optJSONObject("files")?.optJSONObject(name)?.optString("sha256") == want.getString("sha256")) continue
            val part = File(dir, "$name.part")
            val c = get("$baseUrl/$name")
            if (c.responseCode != 200) { log("não consegui baixar $name (${c.responseCode})"); return false }
            val total = want.optLong("bytes", c.contentLengthLong)
            c.inputStream.use { inp ->
                part.outputStream().use { out ->
                    val buf = ByteArray(1 shl 16)
                    var got = 0L
                    var last = 0L
                    while (true) {
                        val n = inp.read(buf); if (n < 0) break
                        out.write(buf, 0, n); got += n
                        if (got - last > (1 shl 20)) { progress(name, got, total); last = got }
                    }
                }
            }
            val sha = Trust.sha256(part)
            if (sha != want.getString("sha256")) {
                part.delete()
                log("$name chegou diferente do que o laboratório selou: recusado")
                return false
            }
            part.renameTo(target)
        }
        File(dir, "manifest.json").writeText(remote.toString(2))
        return true
    }

    /** Organs present on disk whose hash does not match the manifest (nothing is trusted on faith). */
    fun tampered(): List<String> {
        val files = manifest()?.optJSONObject("files") ?: return emptyList()
        return names.filter { n -> File(dir, n).exists() && files.optJSONObject(n)?.optString("sha256")?.let { it != Trust.sha256(File(dir, n)) } == true }
    }
}
