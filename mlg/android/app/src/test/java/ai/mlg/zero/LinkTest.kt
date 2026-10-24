package ai.mlg.zero

import org.json.JSONObject
import java.io.File
import java.util.Base64
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/** The headset's end of the link against the lab's (mlgsub/link.py) on a local broker.
 *  Runs only when tests/test_link.py starts that broker and lab: MLG_TEST_PORT, MLG_TEST_CODE. */
class LinkTest {
    @Test
    fun headsetTalksToTheLab() {
        val port = System.getenv("MLG_TEST_PORT")?.toIntOrNull()
        val code = System.getenv("MLG_TEST_CODE")
        assumeTrue(port != null && code != null)
        val got = LinkedBlockingQueue<Pair<String, JSONObject>>()
        val states = LinkedBlockingQueue<String>()
        val link = Link(code!!, { k, d -> got.put(k to d) }, { states.put(it) }, listOf(Triple("127.0.0.1", port!!, false)))
        link.start()
        assertEquals("online", states.poll(10, TimeUnit.SECONDS))
        assertEquals(Link.topicBase(code), System.getenv("MLG_TEST_TOPIC"))
        link.publish("hello", JSONObject().put("device", "jvm test"))
        link.publish("body", JSONObject().put("decode_ms", 4.2).put("prefill_ms", 0.3).put("threads", 4).put("active_params", 1e6))
        link.publish("ask", JSONObject().put("id", "t1").put("prompt", "U: oi\nM:").put("max", 24))
        var mind = 0
        var answer = ""
        var answered = false
        var transfer: Transfer? = null
        var verified = false
        val file = File.createTempFile("core-link", ".part")
        val end = System.currentTimeMillis() + 40_000
        while (System.currentTimeMillis() < end && !(answered && verified)) {
            transfer?.let { if (it.stalled(3_000)) it.ask() }  // what Mind.watchTransfer does
            val (k, d) = got.poll(1, TimeUnit.SECONDS) ?: continue
            if (k == "mind") mind += d.getJSONArray("items").length()
            if (k == "say" && d.optString("id") == "t1") {
                answer += d.optString("chunk")
                if (d.optBoolean("done")) answered = true
            }
            if (k == "offer" && transfer == null)
                transfer = Transfer(d, file) { from, count ->
                    link.publish("pull", JSONObject().put("sha256", d.getString("sha256")).put("from", from).put("count", count))
                }.also { it.ask() }
            if (k == "part" && transfer?.part(d.getInt("i"), Base64.getDecoder().decode(d.getString("data"))) == true)
                verified = transfer!!.verified()
        }
        link.stop()
        println("mind items: $mind, answer: ${answer.length} chars, core over the link: ${transfer?.received} bytes, verified $verified")
        assertTrue("the lab's living mind reached the headset", mind > 0)
        assertTrue("the lab brain answered", answer.isNotEmpty())
        assertTrue("a core crossed the link, whole and with the sealed hash", verified)
    }
}
