package ai.mlg.zero

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.util.BitSet

/**
 * A Resident Core coming over the live link (mlgsub/link.py offers it once it
 * passed the behaviour suite): the headset pulls it in windows of parts, writes
 * each where it belongs, asks again for whatever went missing, and keeps the
 * file only if the whole of it hashes to what the sealed offer said.
 */
class Transfer(val offer: JSONObject, val file: File, private val pull: (from: Int, count: Int) -> Unit) {
    val sha: String = offer.getString("sha256")
    val size: Long = offer.getLong("bytes")
    private val chunk = offer.getInt("chunk")
    val parts = ((size + chunk - 1) / chunk).toInt()
    private val have = BitSet(parts)
    private val raf = RandomAccessFile(file, "rw").apply { setLength(size) }
    private var windowEnd = 0
    @Volatile var lastHeard = System.currentTimeMillis()
        private set

    val received: Long @Synchronized get() = minOf(size, have.cardinality().toLong() * chunk)

    /** Pulls the next window, from the first part still missing. */
    @Synchronized
    fun ask() {
        val from = have.nextClearBit(0)
        if (from >= parts) return
        windowEnd = minOf(parts, from + WINDOW)
        lastHeard = System.currentTimeMillis()
        pull(from, windowEnd - from)
    }

    /** One part arrived; true once every part is here (the hash is still to be checked). */
    @Synchronized
    fun part(i: Int, data: ByteArray): Boolean {
        if (i !in 0 until parts || have[i]) return complete()
        val expect = if (i == parts - 1) (size - i.toLong() * chunk).toInt() else chunk
        if (data.size != expect) return false
        raf.seek(i.toLong() * chunk)
        raf.write(data)
        have.set(i)
        lastHeard = System.currentTimeMillis()
        if (complete()) {
            raf.close()
            return true
        }
        if (have.nextClearBit(0) >= windowEnd) ask()
        return false
    }

    @Synchronized
    fun complete() = have.cardinality() == parts

    fun stalled(ms: Long) = !complete() && System.currentTimeMillis() - lastHeard > ms

    fun verified() = complete() && Trust.sha256(file) == sha

    fun close() {
        runCatching { raf.close() }
    }

    companion object {
        const val WINDOW = 8
    }
}
