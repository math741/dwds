package ai.mlg.zero

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/** The procedural expert: exact answers, never guessed. Portuguese numbers and
 *  operations in words or symbols ("nove vezes nove", "7*8", "raiz de 81"). */
object Procedural {
    private val units = mapOf(
        "zero" to 0, "um" to 1, "uma" to 1, "dois" to 2, "duas" to 2, "tres" to 3, "quatro" to 4, "cinco" to 5,
        "seis" to 6, "sete" to 7, "oito" to 8, "nove" to 9, "dez" to 10, "onze" to 11, "doze" to 12, "treze" to 13,
        "quatorze" to 14, "catorze" to 14, "quinze" to 15, "dezesseis" to 16, "dezessete" to 17, "dezoito" to 18,
        "dezenove" to 19, "vinte" to 20, "trinta" to 30, "quarenta" to 40, "cinquenta" to 50, "sessenta" to 60,
        "setenta" to 70, "oitenta" to 80, "noventa" to 90, "cem" to 100, "cento" to 100, "duzentos" to 200,
        "trezentos" to 300, "quatrocentos" to 400, "quinhentos" to 500, "seiscentos" to 600,
        "setecentos" to 700, "oitocentos" to 800, "novecentos" to 900)
    private val ops = listOf("dividido por" to "/", "multiplicado por" to "*", "elevado a" to "^", "vezes" to "*",
        "mais" to "+", "menos" to "-", "x" to "*", "por cento de" to "%of")

    private fun fold(s: String) = java.text.Normalizer.normalize(s.lowercase(), java.text.Normalizer.Form.NFD)
        .replace(Regex("\\p{M}"), "")

    /** "vinte e cinco" -> 25, "mil e duzentos" -> 1200; digits pass through. */
    private fun words(text: String): String {
        val out = StringBuilder()
        var acc = -1L
        var total = 0L
        fun flush() {
            if (acc >= 0 || total > 0) out.append(" ").append(total + maxOf(acc, 0)).append(" ")
            acc = -1; total = 0
        }
        val toks = fold(text).replace(Regex("(\\d),(\\d)"), "$1.$2").split(Regex("\\s+"))
        for ((i, t) in toks.withIndex()) {
            val v = units[t]
            when {
                v != null -> acc = maxOf(acc, 0) + v
                t == "mil" -> { total += maxOf(acc, 1) * 1000; acc = 0 }
                t == "e" && (acc >= 0 || total > 0) && i + 1 < toks.size && (units.containsKey(toks[i + 1]) || toks[i + 1] == "mil") -> {}
                else -> { flush(); out.append(t).append(" ") }
            }
        }
        flush()
        return out.toString().replace(Regex("\\s+"), " ").trim()
    }

    fun format(x: Double): String {
        if (x.isNaN() || x.isInfinite()) return "indefinido"
        if (abs(x - Math.rint(x)) < 1e-9 && abs(x) < 1e15) return "%,d".format(Locale("pt", "BR"), x.toLong())
        return "%,.6f".format(Locale("pt", "BR"), x).trimEnd('0').trimEnd(',')
    }

    /** The answer, or null when the request is not a calculation it can do exactly. */
    fun math(question: String): String? {
        var s = words(question)
        for ((w, o) in ops) s = s.replace(Regex("\\b$w\\b"), " $o ")
        Regex("raiz (quadrada )?de (-?[\\d.]+)").find(s)?.let {
            val v = it.groupValues[2].toDouble()
            return "A raiz quadrada de ${format(v)} é ${format(sqrt(v))}."
        }
        Regex("(-?[\\d.]+) ao quadrado").find(s)?.let {
            val v = it.groupValues[1].toDouble(); return "${format(v)} ao quadrado é ${format(v * v)}."
        }
        Regex("(-?[\\d.]+) ao cubo").find(s)?.let {
            val v = it.groupValues[1].toDouble(); return "${format(v)} ao cubo é ${format(v * v * v)}."
        }
        Regex("(-?[\\d.]+) %of (-?[\\d.]+)").find(s)?.let {
            val a = it.groupValues[1].toDouble(); val b = it.groupValues[2].toDouble()
            return "${format(a)}% de ${format(b)} é ${format(a * b / 100)}."
        }
        Regex("(-?[\\d.]+) graus? (celsius )?(para|em) fahrenheit").find(s)?.let {
            val c = it.groupValues[1].toDouble(); return "${format(c)} °C são ${format(c * 9 / 5 + 32)} °F."
        }
        Regex("(-?[\\d.]+) km em metros").find(s)?.let {
            val v = it.groupValues[1].toDouble(); return "${format(v)} km são ${format(v * 1000)} metros."
        }
        Regex("quantos minutos (tem )?(em )?(-?[\\d.]+) horas?").find(s)?.let {
            val v = it.groupValues[3].toDouble(); return "${format(v)} horas têm ${format(v * 60)} minutos."
        }
        val expr = Regex("[-\\d.\\s+*/^()]+").findAll(s).map { it.value.trim() }
            .filter { it.any(Char::isDigit) && it.any { c -> c in "+-*/^" } }.maxByOrNull { it.length } ?: return null
        val v = try { Expr(expr).parse() } catch (e: Exception) { return null }
        return "${expr.replace("*", "×").replace("/", "÷")} = ${format(v)}"
    }

    /** + - * / ^ and parentheses, with the usual precedence. */
    private class Expr(val s: String) {
        var i = 0
        fun parse(): Double { val v = sum(); skip(); require(i == s.length); return v }
        fun skip() { while (i < s.length && s[i] == ' ') i++ }
        fun sum(): Double {
            var v = prod()
            while (true) { skip(); if (i >= s.length) return v
                when (s[i]) { '+' -> { i++; v += prod() }; '-' -> { i++; v -= prod() }; else -> return v } }
        }
        fun prod(): Double {
            var v = pow()
            while (true) { skip(); if (i >= s.length) return v
                when (s[i]) { '*' -> { i++; v *= pow() }; '/' -> { i++; v /= pow() }; else -> return v } }
        }
        fun pow(): Double { val b = unary(); skip(); return if (i < s.length && s[i] == '^') { i++; b.pow(pow()) } else b }
        fun unary(): Double { skip(); if (i < s.length && s[i] == '-') { i++; return -unary() }; return atom() }
        fun atom(): Double {
            skip()
            if (s[i] == '(') { i++; val v = sum(); skip(); require(s[i] == ')'); i++; return v }
            val st = i
            while (i < s.length && (s[i].isDigit() || s[i] == '.')) i++
            return s.substring(st, i).toDouble()
        }
    }

    fun time(question: String): String {
        val now = Date()
        val br = Locale("pt", "BR")
        val q = fold(question)
        return when {
            "hora" in q -> "Agora são ${SimpleDateFormat("HH:mm", br).format(now)}."
            "semana" in q -> "Hoje é ${SimpleDateFormat("EEEE", br).format(now)}."
            "mes" in q -> "Estamos em ${SimpleDateFormat("MMMM 'de' yyyy", br).format(now)}."
            "ano" in q -> "Estamos em ${SimpleDateFormat("yyyy", br).format(now)}."
            else -> "Hoje é ${SimpleDateFormat("EEEE, d 'de' MMMM 'de' yyyy", br).format(now)}, ${SimpleDateFormat("HH:mm", br).format(now)}."
        }
    }
}
