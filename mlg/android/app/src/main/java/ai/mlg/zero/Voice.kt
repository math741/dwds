package ai.mlg.zero

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale

/**
 * Ears and mouth. They are the operating system's speech services, not part of
 * the MLG's brain: whatever this headset has installed. When it has none, the
 * MLG says so and the keyboard (whose microphone key dictates) takes over.
 */
class Voice(private val ctx: Context, private val events: (String, String) -> Unit) {
    private var tts: TextToSpeech? = null
    var canSpeak = false; private set
    val canListen: Boolean = SpeechRecognizer.isRecognitionAvailable(ctx)
    private var recognizer: SpeechRecognizer? = null
    var muted = false

    init {
        tts = TextToSpeech(ctx) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val r = tts?.setLanguage(Locale("pt", "BR")) ?: TextToSpeech.LANG_NOT_SUPPORTED
                canSpeak = r != TextToSpeech.LANG_MISSING_DATA && r != TextToSpeech.LANG_NOT_SUPPORTED
                tts?.setPitch(0.92f)
                tts?.setSpeechRate(1.05f)
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(id: String?) = events("speaking", "1")
                    override fun onDone(id: String?) = events("speaking", "0")
                    @Deprecated("old API") override fun onError(id: String?) = events("speaking", "0")
                })
            }
            events("voice", "${if (canSpeak) 1 else 0}${if (canListen) 1 else 0}")
        }
    }

    private var n = 0
    fun say(text: String) {
        if (!canSpeak || muted || text.isBlank()) return
        tts?.speak(text, TextToSpeech.QUEUE_ADD, null, "mlg-${n++}")
    }

    fun hush() = tts?.stop()

    /** Partial results arrive as ("heard_partial", text), the final one as ("heard", text). */
    fun listen() {
        if (!canListen) { events("heard_error", "sem reconhecimento de voz neste corpo"); return }
        hush()
        val r = recognizer ?: SpeechRecognizer.createSpeechRecognizer(ctx).also { recognizer = it }
        r.setRecognitionListener(object : RecognitionListener {
            override fun onResults(b: Bundle?) =
                events("heard", b?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull() ?: "")
            override fun onPartialResults(b: Bundle?) =
                events("heard_partial", b?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull() ?: "")
            override fun onError(error: Int) = events("heard_error", "erro $error")
            override fun onRmsChanged(rms: Float) = events("level", "%.1f".format(Locale.US, rms))
            override fun onReadyForSpeech(p: Bundle?) = events("listening", "1")
            override fun onEndOfSpeech() = events("listening", "0")
            override fun onBeginningOfSpeech() {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onEvent(t: Int, p: Bundle?) {}
        })
        r.startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "pt-BR")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        })
    }

    fun close() {
        recognizer?.destroy()
        tts?.shutdown()
    }
}
