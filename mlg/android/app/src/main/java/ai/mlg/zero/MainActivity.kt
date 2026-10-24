package ai.mlg.zero

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.view.WindowManager
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import org.json.JSONObject

/** The HUD is a three.js page in app/src/main/assets/hud, served from https://mlg.local
 *  (so its ES modules load), talking to the Mind through window.MLG. */
class MainActivity : Activity() {
    private lateinit var web: WebView
    private lateinit var mind: Mind
    @Volatile private var ready = false
    private val early = mutableListOf<String>()

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(saved: Bundle?) {
        super.onCreate(saved)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        web = WebView(this).apply {
            setBackgroundColor(Color.BLACK)
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.mediaPlaybackRequiresUserGesture = false
            webViewClient = object : WebViewClient() {
                override fun shouldInterceptRequest(view: WebView, req: WebResourceRequest): WebResourceResponse? {
                    if (req.url.host != "mlg.local") return null
                    val path = req.url.path?.trimStart('/')?.ifEmpty { "hud/index.html" } ?: return null
                    val mime = when (path.substringAfterLast('.')) {
                        "html" -> "text/html"; "js" -> "text/javascript"; "css" -> "text/css"; "json" -> "application/json"
                        "png" -> "image/png"; "svg" -> "image/svg+xml"; else -> "application/octet-stream"
                    }
                    return runCatching { WebResourceResponse(mime, "utf-8", assets.open(path)) }.getOrNull()
                }

                override fun onPageFinished(view: WebView, url: String) {
                    ready = true
                    synchronized(early) { early.forEach { web.evaluateJavascript(it, null) }; early.clear() }
                }
            }
            addJavascriptInterface(Bridge(), "MLG")
        }
        setContentView(web)
        mind = Mind(this) { kind, data -> send(kind, data) }
        web.loadUrl("https://mlg.local/hud/index.html")
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        mind.start()
    }

    private fun send(kind: String, data: JSONObject) {
        val js = "window.mlg && window.mlg.on(${JSONObject.quote(kind)}, ${data})"
        if (!ready) synchronized(early) { if (!ready) { early += js; return } }
        runOnUiThread { web.evaluateJavascript(js, null) }
    }

    inner class Bridge {
        @JavascriptInterface fun ask(text: String) { if (text.isNotBlank()) mind.ask(text) }
        @JavascriptInterface fun listen() = runOnUiThread { mind.voice.listen() }
        @JavascriptInterface fun mute(on: Boolean) { mind.voice.muted = on; if (on) mind.voice.hush() }
        @JavascriptInterface fun code(): String = mind.code
        @JavascriptInterface fun pair(code: String) = mind.repair(code)
        @JavascriptInterface fun memory(): String = org.json.JSONArray(mind.memory()).toString()
        @JavascriptInterface fun trust(): String = JSONObject().put("events", mind.trust.seq).put("head", mind.trust.head)
            .put("broken", mind.trust.verify()).toString()
        @JavascriptInterface fun voice(): String = JSONObject().put("speak", mind.voice.canSpeak).put("listen", mind.voice.canListen).toString()
    }

    override fun onDestroy() {
        mind.close()
        super.onDestroy()
    }
}
