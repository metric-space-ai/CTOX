package dev.ctox.businessosmobile

import android.content.Context
import com.google.android.play.core.assetpacks.AssetPackManager
import com.google.android.play.core.assetpacks.AssetPackManagerFactory
import com.google.android.play.core.assetpacks.AssetPackStateUpdateListener
import com.google.android.play.core.assetpacks.model.AssetPackStatus
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

@Serializable data class PackFile(val path: String, val size: Long, val sha256: String)
@Serializable data class PackManifest(
    val format: String, @SerialName("pack_id") val packId: String,
    @SerialName("source_revision") val sourceRevision: String,
    @SerialName("app_version") val appVersion: String,
    @SerialName("total_bytes") val totalBytes: Long, val files: List<PackFile>,
)
sealed interface OfficeState {
    data object Idle : OfficeState
    data class AwaitingConsent(val totalBytes: Long) : OfficeState
    data class Downloading(val progress: Double) : OfficeState
    data object Active : OfficeState
    data object Canceled : OfficeState
    data object Offline : OfficeState
    data class Failed(val message: String) : OfficeState
}

interface OfficePackProvider {
    fun fetch(onProgress: (Double) -> Unit): Boolean
    fun read(relativePath: String): ByteArray
    fun cancel()
}

class LocalDebugOfficeProvider(private val context: Context) : OfficePackProvider {
    private val canceled = AtomicBoolean(false)
    override fun fetch(onProgress: (Double) -> Unit): Boolean { canceled.set(false); onProgress(1.0); return true }
    override fun read(relativePath: String): ByteArray {
        if (canceled.get()) error("office pack request canceled")
        return context.assets.open("debug-office-pack/$relativePath").use { it.readBytes() }
    }
    override fun cancel() { canceled.set(true) }
}

class PlayAssetDeliveryProvider(context: Context) : OfficePackProvider {
    private val manager: AssetPackManager = AssetPackManagerFactory.getInstance(context)
    private val canceled = AtomicBoolean(false)

    override fun fetch(onProgress: (Double) -> Unit): Boolean {
        canceled.set(false)
        if (manager.getPackLocation(PACK_NAME) != null) { onProgress(1.0); return true }
        val complete = CountDownLatch(1)
        var succeeded = false
        val listener = AssetPackStateUpdateListener { state ->
            if (state.name() != PACK_NAME) return@AssetPackStateUpdateListener
            val total = state.totalBytesToDownload()
            onProgress(if (total <= 0L) state.transferProgressPercentage() / 100.0 else state.bytesDownloaded().toDouble() / total)
            when (state.status()) {
                AssetPackStatus.COMPLETED -> { succeeded = manager.getPackLocation(PACK_NAME) != null; complete.countDown() }
                AssetPackStatus.FAILED, AssetPackStatus.CANCELED -> complete.countDown()
            }
        }
        manager.registerListener(listener)
        try {
            manager.fetch(listOf(PACK_NAME)).addOnFailureListener { complete.countDown() }
            return complete.await(5, TimeUnit.MINUTES) && succeeded && !canceled.get()
        } finally {
            manager.unregisterListener(listener)
        }
    }

    override fun read(relativePath: String): ByteArray {
        val root = manager.getPackLocation(PACK_NAME)?.assetsPath() ?: error("office pack unavailable")
        val canonicalRoot = File(root).canonicalFile
        val target = File(canonicalRoot, relativePath).canonicalFile
        require(target.path.startsWith(canonicalRoot.path + File.separator))
        return target.readBytes()
    }

    override fun cancel() { canceled.set(true); manager.cancel(listOf(PACK_NAME)) }
    private companion object { const val PACK_NAME = "ctox_office" }
}

class OfficePackCoordinator(
    private val context: Context,
    private val sourceRevision: String,
    private val appVersion: String,
    private val stateChanged: (OfficeState) -> Unit,
) {
    @Volatile private var active = false
    @Volatile private var approved = false
    @Volatile private var latch: CountDownLatch? = null
    private val executor = Executors.newSingleThreadExecutor()
    private val manifest: PackManifest? = runCatching {
        context.assets.open("office-pack-manifest.json").use { Json.decodeFromString<PackManifest>(it.readBytes().decodeToString()) }
    }.getOrNull()
    private val provider: OfficePackProvider? = manifest?.let {
        if (BuildConfig.DEBUG) LocalDebugOfficeProvider(context) else PlayAssetDeliveryProvider(context)
    }

    fun read(relativePath: String): ByteArray {
        if (!active) {
            val pack = manifest ?: run { stateChanged(OfficeState.Offline); throw IllegalStateException("office pack unavailable") }
            if (!validManifest(pack)) { stateChanged(OfficeState.Failed("Office pack revision mismatch")); throw IllegalStateException("office pack revision mismatch") }
            approved = false
            val waiting = CountDownLatch(1)
            latch = waiting
            stateChanged(OfficeState.AwaitingConsent(pack.totalBytes))
            if (!waiting.await(5, TimeUnit.MINUTES) || !approved || !active) throw IllegalStateException("office pack request canceled")
        }
        require(safePath(relativePath))
        return provider?.read(relativePath) ?: throw IllegalStateException("office pack unavailable")
    }

    fun download() {
        val pack = manifest ?: run { stateChanged(OfficeState.Offline); latch?.countDown(); return }
        val selected = provider ?: run { stateChanged(OfficeState.Offline); latch?.countDown(); return }
        if (!validManifest(pack)) { stateChanged(OfficeState.Failed("Office pack revision mismatch")); latch?.countDown(); return }
        stateChanged(OfficeState.Downloading(0.0))
        executor.execute {
            val ok = runCatching {
                selected.fetch { stateChanged(OfficeState.Downloading(it.coerceIn(0.0, 1.0))) } && verify(selected, pack)
            }.getOrDefault(false)
            active = ok
            approved = ok
            stateChanged(if (ok) OfficeState.Active else OfficeState.Failed("Office pack verification failed. Retry."))
            latch?.countDown()
        }
    }

    fun cancel() {
        provider?.cancel()
        active = false
        approved = false
        stateChanged(OfficeState.Canceled)
        latch?.countDown()
    }

    private fun validManifest(pack: PackManifest) =
        pack.format == "ctox.mobile.shell-pack.v1" && pack.packId == "ctox-office" &&
            pack.sourceRevision == sourceRevision && pack.appVersion == appVersion

    private fun verify(selected: OfficePackProvider, pack: PackManifest): Boolean {
        var verified = 0L
        for (file in pack.files) {
            if (!safePath(file.path)) return false
            val bytes = selected.read(file.path)
            if (bytes.size.toLong() != file.size || sha256(bytes) != file.sha256) return false
            verified += file.size
            stateChanged(OfficeState.Downloading(if (pack.totalBytes == 0L) 1.0 else verified.toDouble() / pack.totalBytes))
        }
        return verified == pack.totalBytes
    }

    private fun safePath(path: String) = path.isNotEmpty() && !path.startsWith("/") && !path.split('/').contains("..")
    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}
