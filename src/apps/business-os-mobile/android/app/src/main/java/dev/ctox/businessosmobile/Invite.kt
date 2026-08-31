package dev.ctox.businessosmobile

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.URI
import java.time.Instant

class InviteValidationException(val code: String) : IllegalArgumentException("Pairing invite rejected ($code)")

@Serializable data class InviteUserRaw(val id: String? = null, @SerialName("display_name") val displayName: String? = null, val role: String? = null, @SerialName("is_admin") val isAdmin: Boolean? = null)
@Serializable data class InviteSessionRaw(val authenticated: Boolean? = null, val source: String? = null, @SerialName("capability_token") val capabilityToken: String? = null, @SerialName("capability_expires_at_ms") val capabilityExpiresAtMs: Long? = null, val user: InviteUserRaw? = null)
@Serializable data class InviteRaw(
    val type: String? = null, val version: Int? = null,
    @SerialName("display_name") val displayName: String? = null,
    @SerialName("instance_id") val instanceId: String? = null,
    @SerialName("sync_room") val syncRoom: String? = null,
    @SerialName("native_peer_id") val nativePeerId: String? = null,
    @SerialName("signaling_urls") val signalingUrls: List<String>? = null,
    @SerialName("signaling_room_password") val password: String? = null,
    val transport: String? = null, @SerialName("expires_at") val expiresAt: String? = null,
    @SerialName("data_plane") val dataPlane: String? = null,
    @SerialName("http_bridge_available") val httpBridgeAvailable: Boolean? = null,
    val session: InviteSessionRaw? = null,
)
data class SessionUser(val id: String, val displayName: String, val role: String)
data class ValidatedInvite(
    val displayName: String, val instanceId: String, val syncRoom: String, val nativePeerId: String,
    val signalingUrls: List<String>, val password: String, val expiresAt: Instant,
    val capabilityToken: String, val capabilityExpiresAt: Instant, val sessionUser: SessionUser,
)

object InviteValidator {
    val json = Json { ignoreUnknownKeys = true; explicitNulls = true }

    fun parseMobileLink(raw: String, now: Instant = Instant.now()): ValidatedInvite {
        val input = raw.trim().ifEmpty { fail("empty") }
        val uri = try { URI(input) } catch (_: Exception) { fail("url") }
        if (uri.scheme != "ctox-business-os-mobile") fail("scheme")
        if (uri.host != "pair" || (uri.path.isNotEmpty() && uri.path != "/") || uri.userInfo != null || uri.fragment != null) fail("host")
        val query = uri.rawQuery ?: fail("query")
        if (!query.startsWith("payload=") || query.contains("&") || query.length <= 8) fail("query")
        val payload = query.removePrefix("payload=")
        if (!payload.matches(Regex("^[A-Za-z0-9_-]+$")) || payload.length > 262_144) fail("payload")
        val bytes = try { java.util.Base64.getUrlDecoder().decode(payload) } catch (_: Exception) { fail("payload") }
        return validate(bytes.decodeToString(), now)
    }

    fun validate(text: String, now: Instant = Instant.now()): ValidatedInvite {
        val element = try { json.parseToJsonElement(text) } catch (_: Exception) { fail("json") }
        val objectValue = try { element.jsonObject } catch (_: Exception) { fail("object") }
        if (objectValue["type"]?.jsonPrimitive?.content != "ctox-business-os-invite") fail("type")
        val versionElement = objectValue["version"]?.jsonPrimitive ?: fail("version")
        if (versionElement.isString || versionElement.content != "1") fail("version")
        val invite = try { json.decodeFromString<InviteRaw>(text) } catch (_: Exception) { fail("json") }
        val displayName = required(invite.displayName, "display_name")
        val instanceId = required(invite.instanceId, "instance_id")
        val syncRoom = required(invite.syncRoom, "sync_room")
        if (!syncRoom.startsWith("ctox-business-os:") || syncRoom.length == "ctox-business-os:".length) fail("sync_room")
        val nativePeerId = required(invite.nativePeerId, "native_peer_id")
        val signalingUrls = invite.signalingUrls?.takeIf { it.isNotEmpty() }?.map(::signalingUrl) ?: fail("signaling_urls")
        val password = required(invite.password, "password")
        if (invite.transport != "webrtc") fail("transport")
        val expiresAt = try { Instant.parse(required(invite.expiresAt, "expires_at")) } catch (_: Exception) { fail("expires_at") }
        if (!expiresAt.isAfter(now)) fail("expired")
        if (invite.dataPlane != "rxdb-webrtc") fail("data_plane")
        if (invite.httpBridgeAvailable != false) fail("http_bridge")
        val session = invite.session?.takeIf { it.authenticated == true } ?: fail("session")
        val capability = required(session.capabilityToken, "capability_token")
        val capabilityExpiresAt = session.capabilityExpiresAtMs?.takeIf { it > 0 }?.let(Instant::ofEpochMilli) ?: fail("capability_expired")
        if (!capabilityExpiresAt.isAfter(now)) fail("capability_expired")
        if (capabilityExpiresAt.isAfter(expiresAt)) fail("capability_expiry")
        val user = session.user ?: fail("user")
        val userId = required(user.id, "user_id")
        val userName = required(user.displayName, "user_display_name")
        val role = required(user.role, "user_role")
        if (role !in setOf("chef", "admin", "founder", "user")) fail("user_role")
        return ValidatedInvite(displayName, instanceId, syncRoom, nativePeerId, signalingUrls, password, expiresAt, capability, capabilityExpiresAt, SessionUser(userId, userName, role))
    }

    private fun required(value: String?, code: String): String = value?.trim()?.takeIf { it.isNotEmpty() } ?: fail(code)
    private fun signalingUrl(raw: String): String {
        val uri = try { URI(raw.trim()) } catch (_: Exception) { fail("signaling_url") }
        if (uri.userInfo != null || uri.fragment != null || uri.host.isNullOrBlank()) fail("signaling_url")
        if (uri.scheme == "wss") return uri.toString()
        fail("signaling_url")
    }
    private fun fail(code: String): Nothing = throw InviteValidationException(code)
}
