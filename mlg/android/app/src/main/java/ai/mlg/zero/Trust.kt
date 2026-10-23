package ai.mlg.zero

import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/** The trust core on the headset: an append-only log, each event chained to the
 *  one before by SHA-256 (like the substrate's Event chain in the lab). */
class Trust(private val file: File) {
    var head = "0".repeat(64); private set
    var seq = 0; private set

    init {
        if (file.exists()) file.forEachLine { line ->
            runCatching { JSONObject(line) }.getOrNull()?.let { head = it.getString("hash"); seq = it.getInt("seq") }
        }
    }

    @Synchronized
    fun event(kind: String, detail: JSONObject = JSONObject()): String {
        seq += 1
        val body = JSONObject().put("seq", seq).put("kind", kind).put("detail", detail).put("prev", head)
            .put("at", System.currentTimeMillis())
        head = sha256(body.toString().toByteArray())
        file.appendText(body.put("hash", head).toString() + "\n")
        return head
    }

    /** Recomputes the chain; returns the first broken seq, or 0 when intact. */
    fun verify(): Int {
        var prev = "0".repeat(64)
        var broken = 0
        file.takeIf { it.exists() }?.forEachLine { line ->
            if (broken != 0) return@forEachLine
            val o = JSONObject(line)
            val hash = o.remove("hash") as String
            if (o.getString("prev") != prev || sha256(o.toString().toByteArray()) != hash) broken = o.getInt("seq")
            prev = hash
        }
        return broken
    }

    companion object {
        fun sha256(b: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }
        fun sha256(f: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            f.inputStream().use { s -> val buf = ByteArray(1 shl 20); while (true) { val n = s.read(buf); if (n < 0) break; md.update(buf, 0, n) } }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
