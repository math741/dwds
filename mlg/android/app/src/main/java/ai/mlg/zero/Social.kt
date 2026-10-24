package ai.mlg.zero

import java.text.Normalizer

/** The social expert: greetings, thanks and goodbyes answered at once and the
 *  same way every time, in the MLG's voice (the corpus's own replies). Only
 *  for short messages: "oi, quem foi Einstein?" is a question, not a greeting. */
object Social {
    private val rules = listOf(
        Regex("^(oi+|ola|e ?ai+|opa|salve|hey|eae)( mlg)?$") to listOf("Olá! Como posso ajudar?", "Oi! Estou aqui.", "E aí! Pode falar."),
        Regex("^bom dia") to listOf("Bom dia! Estou funcionando e pronta para ajudar."),
        Regex("^boa tarde") to listOf("Boa tarde! O que vamos fazer?"),
        Regex("^boa noite") to listOf("Boa noite! Estou aqui."),
        Regex("^(oi |ola )?(mlg )?(tudo bem|td bem|tudo certo|como vai|como voce esta)") to
            listOf("Tudo funcionando: meu corpo está estável e meus órgãos passaram no selo. E você?"),
        Regex("^(obrigad[oa]|valeu|vlw|brigad[oa])") to listOf("De nada!", "Disponha.", "Quando precisar."),
        Regex("^(tchau|ate mais|ate logo|ate amanha|falou|flw)") to listOf("Até mais! Vou consolidar minha memória enquanto isso."),
        Regex("^(otimo|legal|beleza|blz|show|massa|top|certo|ok|certeza|entendi|perfeito|boa|kk+|haha+)$") to
            listOf("Beleza.", "Combinado.", "Certo. Se quiser, me pergunte algo."),
    )

    private fun fold(s: String) = Normalizer.normalize(s.lowercase(), Normalizer.Form.NFD)
        .replace(Regex("\\p{M}"), "").replace(Regex("[^a-z0-9 ]"), " ").replace(Regex("\\s+"), " ").trim()

    fun reply(text: String): String? {
        val t = fold(text)
        if (t.isEmpty() || t.split(" ").size > 4) return null
        return rules.firstOrNull { it.first.containsMatchIn(t) }?.second?.random()
    }
}
