package ai.mlg.zero

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.Normalizer

/** Persistent memory, the headset's "partição de estado": what you told the MLG
 *  to keep, recalled by the words you ask with. One JSON file, written whole. */
class Memory(private val file: File) {
    data class Fact(val text: String, val at: Long, val key: String?)

    private val facts = mutableListOf<Fact>()

    init {
        if (file.exists()) runCatching {
            val a = JSONArray(file.readText())
            for (i in 0 until a.length()) a.getJSONObject(i).let {
                facts += Fact(it.getString("text"), it.getLong("at"), it.optString("key").ifEmpty { null })
            }
        }
    }

    val size get() = facts.size

    private fun save() {
        val a = JSONArray()
        for (f in facts) a.put(JSONObject().put("text", f.text).put("at", f.at).put("key", f.key ?: ""))
        val tmp = File(file.path + ".tmp")
        tmp.writeText(a.toString())
        tmp.renameTo(file)
    }

    private fun fold(s: String) = Normalizer.normalize(s.lowercase(), Normalizer.Form.NFD).replace(Regex("\\p{M}"), "")

    /** Keeps what the user said, minus the "lembra que" in front; returns what was kept. */
    fun remember(said: String): String {
        var t = said.trim().trimEnd('.', '!')
        t = t.replace(Regex("^(por favor,? )?(lembr[ae]|anota|guarda( isso)?|memoriza|n[aã]o esque[cç]a)( de| que| isso)?[:,]?\\s*",
            RegexOption.IGNORE_CASE), "")
        val f = fold(t)
        val key = Regex("^(meu|minha) ([a-z ]+?) (e|eh|é) ").find(f)?.groupValues?.get(2)
            ?: Regex("^eu me chamo").find(f)?.let { "nome" }
        if (key != null) facts.removeAll { it.key == key }  // a new name replaces the old one
        facts += Fact(t, System.currentTimeMillis(), key)
        save()
        return t
    }

    /** The facts that share the most words with the question, best first. */
    fun recall(question: String, top: Int = 3): List<Fact> {
        val q = fold(question).split(Regex("[^a-z0-9]+")).filter { it.length > 2 }.toSet()
        val wantsName = Regex("\\b(nome|chamo|quem eu sou)\\b").containsMatchIn(fold(question))
        return facts.map { f ->
            val w = fold(f.text).split(Regex("[^a-z0-9]+")).toSet()
            f to (q.count { it in w } + if (wantsName && f.key == "nome") 3 else 0)
        }.filter { it.second > 0 }.sortedWith(compareBy({ -it.second }, { -it.first.at })).take(top).map { it.first }
    }

    fun all() = facts.toList()
}
