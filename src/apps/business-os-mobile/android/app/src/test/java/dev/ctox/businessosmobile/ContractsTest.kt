package dev.ctox.businessosmobile

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.security.MessageDigest
import java.time.Instant
import kotlin.io.path.createTempDirectory

class ContractsTest {
    private val corpus = javaClass.classLoader!!.getResourceAsStream("invites.json")!!.use { Json.parseToJsonElement(it.readBytes().decodeToString()).jsonObject }
    private val now = Instant.parse(corpus["now"]!!.jsonPrimitive.content)
    private val valid = corpus["valid"]!!.jsonObject

    @Test fun validInviteAndEverySharedRejection() {
        val parsed = InviteValidator.validate(valid.toString(), now)
        assertEquals(valid["instance_id"]!!.jsonPrimitive.content, parsed.instanceId)
        corpus["rejections"]!!.jsonArray.forEach { element ->
            val rejection = element.jsonObject
            val candidate = setPath(valid, rejection["path"]!!.jsonPrimitive.content.split('.'), rejection["value"]!!)
            try { InviteValidator.validate(candidate.toString(), now); fail("expected ${rejection["code"]}") }
            catch (error: InviteValidationException) { assertEquals(rejection["code"]!!.jsonPrimitive.content, error.code) }
        }
    }

    @Test fun atomicRepairForgetAndOpaqueRegistry() {
        val invite = InviteValidator.validate(valid.toString(), now)
        val registry = MemoryRegistry(); val secrets = MemorySecrets(); val repository = PairingRepository(registry, secrets)
        val first = repository.pair(invite)
        assertEquals(invite.password, secrets.get(first.passwordRef))
        assertFalse(Json.encodeToString(registry.value).contains(invite.password))
        val second = repository.pair(invite)
        assertEquals(first.profileName, second.profileName)
        assertNull(secrets.get(first.passwordRef))
        var removed: String? = null
        repository.forget(second.id) { removed = it }
        assertEquals(second.profileName, removed); assertTrue(registry.value.instances.isEmpty())
    }

    @Test fun failedRepairLeavesPreviousRegistryAndSecrets() {
        val invite = InviteValidator.validate(valid.toString(), now)
        val registry = MemoryRegistry(); val secrets = MemorySecrets(); val repository = PairingRepository(registry, secrets)
        val first = repository.pair(invite); secrets.failAt = secrets.writes + 2
        assertThrows(Exception::class.java) { repository.pair(invite) }
        assertEquals(first.passwordRef, registry.value.instances.single().passwordRef)
        assertEquals(invite.password, secrets.get(first.passwordRef))
    }

    @Test fun launchNavigationProfileAndOfficeContracts() {
        assertThrows(IllegalArgumentException::class.java) { requireMultiProfile(false) }
        assertEquals(NavigationDecision.ALLOW, navigationDecision("https://appassets.androidplatform.net/business-os/index.html"))
        assertEquals(NavigationDecision.EXTERNAL, navigationDecision("https://docs.example.test/help"))
        assertEquals(NavigationDecision.DENY, navigationDecision("data:text/html,no"))
        val invite = InviteValidator.validate(valid.toString(), now)
        val instance = PairingRepository(MemoryRegistry(), MemorySecrets()).pair(invite)
        val launch = LaunchContextBuilder.build(instance, invite.password, invite.capabilityToken)
        val injected = LaunchContextBuilder.inject("<html><head><script src='first.js'></script></head></html>".encodeToByteArray(), launch.first, launch.second).decodeToString()
        assertTrue(injected.indexOf("data-ctox-mobile-bootstrap") < injected.indexOf("first.js"))
        val hostileInstance = instance.copy(sessionUser = instance.sessionUser.copy(displayName = "</script><script>throw 1</script>"))
        val hostileLaunch = LaunchContextBuilder.build(hostileInstance, invite.password, invite.capabilityToken)
        val hardened = LaunchContextBuilder.inject("<html><head></head></html>".encodeToByteArray(), hostileLaunch.first, hostileLaunch.second).decodeToString()
        assertFalse(hardened.contains("</script><script>throw 1</script>"))
        assertTrue(hardened.contains("\\u003c/script\\u003e"))
        val root = createTempDirectory("ctox-office-").toFile(); val file = File(root, "office.bin").apply { writeText("office") }
        val hash = MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") { "%02x".format(it) }
        val manifest = PackManifest("ctox.mobile.shell-pack.v1", "ctox-office", "rev", "0.1.0", file.length(), listOf(PackFile("office.bin", file.length(), hash)))
        var progress = 0.0; PortablePackVerifier.verify(root, manifest, "rev", "0.1.0", progress = { progress = it }); assertEquals(1.0, progress, 0.0)
        assertThrows(IllegalArgumentException::class.java) { PortablePackVerifier.verify(root, manifest, "stale", "0.1.0") }
        assertThrows(IllegalStateException::class.java) { PortablePackVerifier.verify(root, manifest, "rev", "0.1.0", canceled = { true }) }
        file.writeText("corrupt")
        assertThrows(IllegalArgumentException::class.java) { PortablePackVerifier.verify(root, manifest, "rev", "0.1.0") }
    }

    private fun setPath(root: JsonObject, parts: List<String>, value: JsonElement): JsonObject = buildJsonObject {
        root.forEach { (key, current) -> put(key, if (key == parts.first()) { if (parts.size == 1) value else setPath(current.jsonObject, parts.drop(1), value) } else current) }
    }
}
private class MemoryRegistry : RegistryStore { var value = MobileRegistry(); override fun load() = value; override fun save(registry: MobileRegistry) { RegistrySafety.check(registry); value = registry } }
private class MemorySecrets : SecretStore {
    val values = mutableMapOf<String, String>(); var writes = 0; var failAt = 0
    override fun set(ref: String, value: String) { writes++; if (writes == failAt) error("synthetic write failure"); values[ref] = value }
    override fun get(ref: String) = values[ref]
    override fun delete(ref: String) { values.remove(ref) }
}
