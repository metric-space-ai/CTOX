package dev.ctox.businessosmobile

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.webkit.GeolocationPermissions
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.webkit.ProfileStore
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewCompat
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.ByteArrayInputStream

object LaunchContextBuilder {
    private fun scriptSafeJson(value: String): String = value
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")

    fun build(instance: MobileInstance, password: String, capability: String): Pair<String, String> {
        require(password.isNotEmpty() && capability.isNotEmpty() && instance.capabilityExpiresAtMs > System.currentTimeMillis())
        val session = buildJsonObject {
            put("authenticated", true); put("source", "android_invite"); put("capability_token", capability); put("capability_expires_at_ms", instance.capabilityExpiresAtMs)
            put("user", buildJsonObject { put("id", instance.sessionUser.id); put("display_name", instance.sessionUser.displayName); put("role", instance.sessionUser.role); put("is_admin", instance.sessionUser.role in setOf("chef", "admin", "founder")) })
        }
        val config = buildJsonObject {
            put("instance_id", instance.instanceId); put("peer_id", "android:${instance.id}"); put("peer_role", "business_os_client"); put("native_peer_id", instance.nativePeerId)
            put("sync_room", instance.syncRoom); put("signaling_urls", buildJsonArray { instance.signalingUrls.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
            put("signaling_room_password", password); put("transport", "webrtc"); put("data_plane", "rxdb-webrtc"); put("http_bridge_available", false)
            put("app_hosting", "android_bundled_shell"); put("ctox_instance_required", true); put("session", session)
        }
        return scriptSafeJson(session.toString()) to scriptSafeJson(config.toString())
    }
    fun inject(html: ByteArray, session: String, config: String): ByteArray {
        val text = html.decodeToString(); val match = Regex("<head(?:\\s[^>]*)?>", RegexOption.IGNORE_CASE).find(text) ?: error("shell index missing head")
        val hardening = "try{Object.defineProperty(navigator,'clipboard',{value:{read:()=>Promise.reject(new DOMException('Denied','NotAllowedError')),readText:()=>Promise.reject(new DOMException('Denied','NotAllowedError')),write:()=>Promise.reject(new DOMException('Denied','NotAllowedError')),writeText:()=>Promise.reject(new DOMException('Denied','NotAllowedError'))},configurable:false})}catch(_){}"
        val script = "<script data-ctox-mobile-bootstrap>window.CTOX_BUSINESS_OS_SESSION=$session;window.CTOX_BUSINESS_OS_CONFIG=$config;window.CTOX_BUSINESS_OS_DESIGN_TEMPLATES=[];$hardening</script>"
        return (text.substring(0, match.range.last + 1) + script + text.substring(match.range.last + 1)).encodeToByteArray()
    }
}

class MobileAssetHandler(
    private val activity: Activity,
    private val session: String,
    private val config: String,
    private val office: OfficePackCoordinator,
) : WebViewAssetLoader.PathHandler {
    override fun handle(path: String): WebResourceResponse? = try {
        val clean = path.ifEmpty { "index.html" }
        if (clean.startsWith("vendor/ctox-office/")) {
            val bytes = office.read(clean.removePrefix("vendor/ctox-office/"))
            WebResourceResponse(mime(clean), null, 200, "OK", mapOf("Cache-Control" to "public, max-age=31536000, immutable", "X-Content-Type-Options" to "nosniff"), ByteArrayInputStream(bytes))
        } else {
            require(!clean.split('/').contains(".."))
            val raw = activity.assets.open("business-os/$clean").use { it.readBytes() }
            val bytes = if (clean == "index.html") LaunchContextBuilder.inject(raw, session, config) else raw
            val cache = if (clean == "index.html") "no-store" else "public, max-age=31536000, immutable"
            WebResourceResponse(mime(clean), null, 200, "OK", mapOf("Cache-Control" to cache, "X-Content-Type-Options" to "nosniff"), ByteArrayInputStream(bytes))
        }
    } catch (_: Exception) { null }
    private fun mime(path: String) = when (path.substringAfterLast('.', "").lowercase()) { "html" -> "text/html"; "css" -> "text/css"; "js", "mjs" -> "text/javascript"; "json" -> "application/json"; "svg" -> "image/svg+xml"; "png" -> "image/png"; "jpg", "jpeg" -> "image/jpeg"; "wasm" -> "application/wasm"; "woff" -> "font/woff"; "woff2" -> "font/woff2"; else -> "application/octet-stream" }
}

fun createBusinessOsWebView(activity: Activity, instance: MobileInstance, password: String, capability: String, profileStore: ProfileStore, office: OfficePackCoordinator): WebView {
    val (session, config) = LaunchContextBuilder.build(instance, password, capability)
    val loader = WebViewAssetLoader.Builder().setDomain("appassets.androidplatform.net").addPathHandler("/business-os/", MobileAssetHandler(activity, session, config, office)).build()
    return WebView(activity).apply {
        profileStore.getOrCreateProfile(instance.profileName)
        WebViewCompat.setProfile(this, instance.profileName)
        settings.javaScriptEnabled = true; settings.domStorageEnabled = true
        settings.allowFileAccess = false; settings.allowContentAccess = false; settings.setGeolocationEnabled(false)
        settings.mediaPlaybackRequiresUserGesture = true; settings.mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
        settings.javaScriptCanOpenWindowsAutomatically = false; settings.setSupportMultipleWindows(false)
        webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) { request.deny() }
            override fun onGeolocationPermissionsShowPrompt(origin: String?, callback: GeolocationPermissions.Callback) { callback.invoke(origin, false, false) }
        }
        webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? = loader.shouldInterceptRequest(request.url)
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                val uri = request.url
                if (uri.scheme == "https" && uri.host == "appassets.androidplatform.net" && uri.path?.startsWith("/business-os/") == true) return false
                if (uri.scheme == "https" && request.isForMainFrame && request.hasGesture()) activity.startActivity(Intent(Intent.ACTION_VIEW, uri))
                return true
            }
        }
        loadUrl("https://appassets.androidplatform.net/business-os/index.html")
    }
}
