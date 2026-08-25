package dev.ctox.businessosmobile

import android.content.Context
import android.util.AtomicFile
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.security.MessageDigest
import java.util.UUID

@Serializable data class RegistryUser(val id: String, val displayName: String, val role: String)
@Serializable data class MobileInstance(
    val id: String, val displayName: String, val instanceId: String, val syncRoom: String,
    val nativePeerId: String, val signalingUrls: List<String>, val expiresAt: String,
    val capabilityExpiresAtMs: Long, val sessionUser: RegistryUser,
    val passwordRef: String, val capabilityRef: String, val profileName: String,
)
@Serializable data class MobileRegistry(val version: Int = 1, val instances: List<MobileInstance> = emptyList())

interface SecretStore { fun set(ref: String, value: String); fun get(ref: String): String?; fun delete(ref: String) }
interface RegistryStore { fun load(): MobileRegistry; fun save(registry: MobileRegistry) }

class FileRegistryStore(context: Context) : RegistryStore {
    private val file = AtomicFile(File(context.noBackupFilesDir, "instances-v1.json"))
    override fun load(): MobileRegistry = try { Json.decodeFromString(file.readFully().decodeToString()) } catch (_: java.io.FileNotFoundException) { MobileRegistry() }
    override fun save(registry: MobileRegistry) {
        RegistrySafety.check(registry)
        val output = file.startWrite()
        try { output.write(Json.encodeToString(registry).encodeToByteArray()); file.finishWrite(output) }
        catch (error: Throwable) { file.failWrite(output); throw error }
    }
}

object RegistrySafety {
    fun check(registry: MobileRegistry) {
        val text = Json.encodeToString(registry)
        listOf("signaling_room_password", "capability_token", "ctox_config", "payload=").forEach { require(!text.contains(it)) }
    }
}

class PairingRepository(private val registry: RegistryStore, private val secrets: SecretStore) {
    @Synchronized fun pair(invite: ValidatedInvite): MobileInstance {
        val state = registry.load()
        val previous = state.instances.firstOrNull { it.instanceId == invite.instanceId }
        val id = previous?.id ?: "paired:${sha256(invite.instanceId).take(24)}"
        val generation = UUID.randomUUID().toString()
        val profile = previous?.profileName ?: "ctox_${UUID.randomUUID().toString().replace("-", "")}"
        val next = MobileInstance(
            id, invite.displayName, invite.instanceId, invite.syncRoom, invite.nativePeerId,
            invite.signalingUrls, invite.expiresAt.toString(), invite.capabilityExpiresAt.toEpochMilli(),
            RegistryUser(invite.sessionUser.id, invite.sessionUser.displayName, invite.sessionUser.role),
            "keystore://ctox-business-os-mobile/$id/$generation/room",
            "keystore://ctox-business-os-mobile/$id/$generation/capability",
            profile,
        )
        val written = mutableListOf<String>()
        try {
            secrets.set(next.passwordRef, invite.password); written += next.passwordRef
            secrets.set(next.capabilityRef, invite.capabilityToken); written += next.capabilityRef
            registry.save(MobileRegistry(instances = state.instances.filterNot { it.instanceId == invite.instanceId } + next))
        } catch (error: Throwable) { written.forEach { runCatching { secrets.delete(it) } }; throw error }
        previous?.let { runCatching { secrets.delete(it.passwordRef) }; runCatching { secrets.delete(it.capabilityRef) } }
        return next
    }

    @Synchronized fun forget(id: String, deleteProfile: (String) -> Unit) {
        val state = registry.load(); val target = state.instances.firstOrNull { it.id == id } ?: return
        registry.save(MobileRegistry(instances = state.instances.filterNot { it.id == id }))
        runCatching { secrets.delete(target.passwordRef) }; runCatching { secrets.delete(target.capabilityRef) }
        deleteProfile(target.profileName)
    }
    private fun sha256(value: String) = MessageDigest.getInstance("SHA-256").digest(value.encodeToByteArray()).joinToString("") { "%02x".format(it) }
}
