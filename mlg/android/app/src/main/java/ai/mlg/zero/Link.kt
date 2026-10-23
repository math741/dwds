package ai.mlg.zero

import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.OutputStream
import java.net.InetSocketAddress
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec
import javax.net.ssl.SSLSocketFactory

/**
 * The headset's end of the live link (mlgsub/link.py is the lab's end): MQTT
 * 3.1.1 over TLS to a public broker, every payload AES-GCM sealed with a key
 * derived from the pairing code, topics hashed from it. Only what the MLG
 * needs of MQTT: connect, subscribe, publish at QoS 0, ping.
 */
class Link(code: String, private val onMessage: (String, JSONObject) -> Unit, private val onState: (String) -> Unit) {
    companion object {
        val BROKERS = listOf("broker.emqx.io" to 8883, "broker.hivemq.com" to 8883, "test.mosquitto.org" to 8886)
        const val ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

        fun newCode(): String {
            val r = SecureRandom()
            return (0 until 4).joinToString("-") { (0 until 4).map { ALPHABET[r.nextInt(ALPHABET.length)] }.joinToString("") }
        }

        fun norm(code: String) = code.uppercase().filter { it in ALPHABET }
        private fun sha(s: String) = MessageDigest.getInstance("SHA-256").digest(s.toByteArray())
        fun topicBase(code: String) = "mlg/" + sha("mlg-link-topic:" + norm(code)).joinToString("") { "%02x".format(it) }.take(24)
    }

    private val base = topicBase(code)
    private val key = SecretKeySpec(sha("mlg-link-key:" + norm(code)), "AES")
    private val random = SecureRandom()
    @Volatile private var out: OutputStream? = null
    @Volatile var connected = false; private set
    @Volatile var broker = ""; private set
    @Volatile private var running = true

    fun start() = Thread({ loop() }, "mlg-link").apply { isDaemon = true; start() }

    fun stop() {
        running = false
        runCatching { out?.close() }
    }

    fun publish(kind: String, obj: JSONObject) {
        val o = out ?: return
        val nonce = ByteArray(12).also(random::nextBytes)
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, key, GCMParameterSpec(128, nonce)) }
        val payload = nonce + c.doFinal(obj.toString().toByteArray())
        val topic = "$base/$kind".toByteArray()
        val body = ByteArrayOutputStream().apply { str(topic); write(payload) }.toByteArray()
        runCatching { synchronized(this) { packet(o, 0x30, body) } }
    }

    private fun open(blob: ByteArray): JSONObject? = runCatching {
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.DECRYPT_MODE, key, GCMParameterSpec(128, blob.copyOfRange(0, 12)))
        }
        JSONObject(String(c.doFinal(blob, 12, blob.size - 12)))
    }.getOrNull()

    @Volatile private var wasOnline = false

    /** Brokers are tried in the lab's order, always from the first, so both ends meet on the same one. */
    private fun loop() {
        var attempt = 0
        while (running) {
            for ((host, port) in BROKERS) {
                if (!running) return
                wasOnline = false
                try {
                    session(host, port)
                } catch (e: Exception) {
                    connected = false
                    onState("offline")
                }
                if (wasOnline) break
            }
            if (wasOnline) attempt = 0 else Thread.sleep(minOf(30_000L, 2_000L shl minOf(attempt++, 4)))
        }
    }

    private fun session(host: String, port: Int) {
        val sock = SSLSocketFactory.getDefault().createSocket().apply { connect(InetSocketAddress(host, port), 8000); soTimeout = 45_000 }
        sock.use { s ->
            val o = s.getOutputStream()
            val inp = DataInputStream(s.getInputStream().buffered())
            val id = "mlg-quest-" + (0 until 8).map { "0123456789abcdef"[random.nextInt(16)] }.joinToString("")
            packet(o, 0x10, ByteArrayOutputStream().apply {
                str("MQTT".toByteArray()); write(4); write(0x02); write(0); write(30); str(id.toByteArray())
            }.toByteArray())
            val (type, ack) = read(inp)
            if (type != 0x20 || ack.size < 2 || ack[1].toInt() != 0) error("refused")
            packet(o, 0x82, ByteArrayOutputStream().apply {
                write(0); write(1); str("$base/#".toByteArray()); write(0)
            }.toByteArray())
            out = o
            connected = true
            wasOnline = true
            broker = host
            onState("online")
            val pinger = Thread {
                while (connected && out === o) {
                    Thread.sleep(20_000)
                    runCatching { synchronized(this) { packet(o, 0xC0, ByteArray(0)) } }
                }
            }.apply { isDaemon = true; start() }
            while (running) {
                val (t, body) = read(inp)
                if (t and 0xF0 != 0x30) continue
                val tl = ((body[0].toInt() and 0xFF) shl 8) or (body[1].toInt() and 0xFF)
                val topic = String(body, 2, tl)
                val qos = (t shr 1) and 3
                val start = 2 + tl + if (qos > 0) 2 else 0
                val kind = topic.substringAfterLast('/')
                if (kind in setOf("ask", "hello", "body")) continue  // our own messages, echoed
                open(body.copyOfRange(start, body.size))?.let { onMessage(kind, it) }
            }
            pinger.interrupt()
        }
        out = null
        connected = false
    }

    private fun ByteArrayOutputStream.str(b: ByteArray) { write(b.size shr 8); write(b.size and 0xFF); write(b) }

    private fun packet(o: OutputStream, type: Int, body: ByteArray) {
        val head = ByteArrayOutputStream()
        head.write(type)
        var n = body.size
        do { var d = n % 128; n /= 128; if (n > 0) d = d or 0x80; head.write(d) } while (n > 0)
        o.write(head.toByteArray() + body)
        o.flush()
    }

    private fun read(inp: DataInputStream): Pair<Int, ByteArray> {
        val type = inp.readUnsignedByte()
        var mult = 1
        var len = 0
        while (true) {
            val d = inp.readUnsignedByte()
            len += (d and 0x7F) * mult
            if (d and 0x80 == 0) break
            mult *= 128
        }
        val b = ByteArray(len)
        inp.readFully(b)
        return type to b
    }
}
