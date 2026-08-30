package dev.ctox.businessosmobile

import java.io.File
import java.security.MessageDigest

fun requireMultiProfile(supported: Boolean) { require(supported) { "Android WebView MULTI_PROFILE is required" } }
enum class NavigationDecision { ALLOW, EXTERNAL, DENY }
fun navigationDecision(raw: String): NavigationDecision = try {
    val uri = java.net.URI(raw)
    if (uri.scheme == "https" && uri.host == "appassets.androidplatform.net" && uri.path.startsWith("/business-os/")) NavigationDecision.ALLOW
    else if (uri.scheme == "https") NavigationDecision.EXTERNAL else NavigationDecision.DENY
} catch (_: Exception) { NavigationDecision.DENY }

object PortablePackVerifier {
    fun verify(root: File, manifest: PackManifest, sourceRevision: String, appVersion: String, canceled: () -> Boolean = { false }, progress: (Double) -> Unit = {}) {
        require(manifest.format == "ctox.mobile.shell-pack.v1" && manifest.packId == "ctox-office")
        require(manifest.sourceRevision == sourceRevision) { "office pack revision mismatch" }
        require(manifest.appVersion == appVersion) { "office pack app version mismatch" }
        var verified = 0L
        manifest.files.forEach { entry ->
            check(!canceled()) { "office pack activation canceled" }
            require(!entry.path.startsWith("/") && !entry.path.split('/').contains(".."))
            val file = File(root, entry.path).canonicalFile
            require(file.path.startsWith(root.canonicalPath + File.separator) && file.isFile)
            val bytes = file.readBytes(); require(bytes.size.toLong() == entry.size)
            val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
            require(hash == entry.sha256) { "office pack hash mismatch" }
            verified += entry.size; progress(if (manifest.totalBytes == 0L) 1.0 else verified.toDouble() / manifest.totalBytes)
        }
        require(verified == manifest.totalBytes)
    }
}
