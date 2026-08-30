package dev.ctox.businessosmobile

import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.text.InputType
import android.text.TextUtils
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.webkit.ProfileStore
import androidx.webkit.WebViewFeature
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.time.ZoneId
import java.time.format.DateTimeFormatter

class MainActivity : ComponentActivity() {
    private lateinit var registryStore: FileRegistryStore
    private lateinit var secretStore: KeystoreSecretStore
    private lateinit var repository: PairingRepository
    private lateinit var profileStore: ProfileStore
    private lateinit var root: FrameLayout
    private var webView: WebView? = null
    private var multiProfileSupported = false
    private val scanner = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        result.data?.getStringExtra(ScannerActivity.RESULT_VALUE)?.let { reviewLink(it, false) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        multiProfileSupported = WebViewFeature.isFeatureSupported(WebViewFeature.MULTI_PROFILE)
        if (!multiProfileSupported) { setContentView(unsupportedView()); return }
        profileStore = ProfileStore.getInstance()
        registryStore = FileRegistryStore(this); secretStore = KeystoreSecretStore(this); repository = PairingRepository(registryStore, secretStore)
        root = FrameLayout(this)
        setContentView(root)
        showInstances()
        intent?.dataString?.let { reviewLink(it, false) }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        if (multiProfileSupported) intent.dataString?.let { reviewLink(it, false) }
    }
    override fun onDestroy() { webView?.destroy(); super.onDestroy() }

    private fun dp(value: Int): Int = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value.toFloat(), resources.displayMetrics).toInt()

    private fun boundedContentWidth(): Int = dp(minOf(resources.configuration.screenWidthDp, 720))


    private fun secondaryColor(): Int = ContextCompat.getColor(this, R.color.ctox_text_secondary)

    private fun label(textRes: Int, sizeSp: Float, bold: Boolean = false, secondary: Boolean = false): TextView =
        TextView(this).apply {
            setText(textRes)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp)
            if (bold) setTypeface(typeface, Typeface.BOLD)
            if (secondary) setTextColor(secondaryColor())
        }

    private fun unsupportedView(): View {
        val scroll = ScrollView(this).apply { isFillViewport = true }
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(dp(32), dp(24), dp(32), dp(24))
        }
        column.addView(label(R.string.unsupported_title, 20f, bold = true).apply { gravity = Gravity.CENTER })
        column.addView(label(R.string.unsupported_body, 16f, secondary = true).apply {
            gravity = Gravity.CENTER
            setPadding(0, dp(12), 0, 0)
        })
        scroll.addView(column, FrameLayout.LayoutParams(dp(minOf(resources.configuration.screenWidthDp, 640)), ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.CENTER))
        return scroll
    }

    private fun showInstances() {
        webView?.destroy(); webView = null; root.removeAllViews()
        val scroll = ScrollView(this)
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24), dp(16), dp(24), dp(24))
        }
        scroll.addView(column, FrameLayout.LayoutParams(boundedContentWidth(), ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.CENTER_HORIZONTAL))
        root.addView(scroll, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        column.addView(label(R.string.app_name, 26f, bold = true).apply { setPadding(0, dp(12), 0, dp(4)) })
        column.addView(label(R.string.instances_subtitle, 15f, secondary = true).apply { setPadding(0, 0, 0, dp(20)) })
        column.addView(Button(this).apply {
            setText(R.string.add_pair_instance)
            minHeight = dp(48)
            setOnClickListener { pairingDialog() }
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { bottomMargin = dp(16) })
        val instances = runCatching { registryStore.load().instances }.getOrElse { emptyList() }
        if (instances.isEmpty()) {
            column.addView(label(R.string.no_instances, 17f, secondary = true).apply { setPadding(0, dp(24), 0, dp(8)) })
        }
        instances.forEach { instance -> column.addView(instanceRow(instance), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { bottomMargin = dp(12) }) }
    }

    private fun instanceRow(instance: MobileInstance): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.instance_row_bg)
            setPadding(dp(16), dp(16), dp(16), dp(12))
        }
        row.addView(TextView(this).apply {
            text = instance.displayName
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
            setTypeface(typeface, Typeface.BOLD)
        })
        row.addView(TextView(this).apply {
            text = instance.instanceId
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setTextColor(secondaryColor())
            setPadding(0, dp(2), 0, dp(12))
        })
        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        actions.addView(Button(this).apply {
            setText(R.string.open)
            minHeight = dp(48)
            contentDescription = getString(R.string.open_instance_cd, instance.displayName)
            setOnClickListener { openInstance(instance) }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(6) })
        actions.addView(Button(this).apply {
            setText(R.string.forget)
            minHeight = dp(48)
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ctox_destructive))
            contentDescription = getString(R.string.forget_instance_cd, instance.displayName)
            setOnClickListener { confirmForget(instance) }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(6) })
        row.addView(actions, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        return row
    }

    private fun pairingDialog() {
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(4), dp(20), dp(4))
        }
        val field = EditText(this).apply {
            setHint(R.string.pairing_link_hint)
            minLines = 3
            maxLines = 6
            isVerticalScrollBarEnabled = true
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        }
        layout.addView(field, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        val quick = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val pasteButton = Button(this).apply { setText(R.string.paste); minHeight = dp(48) }
        val scanButton = Button(this).apply { setText(R.string.scan_qr); minHeight = dp(48) }
        quick.addView(pasteButton, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(6); topMargin = dp(12) })
        quick.addView(scanButton, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(6); topMargin = dp(12) })
        layout.addView(quick, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.add_instance)
            .setView(layout)
            .setPositiveButton(R.string.review) { _, _ -> reviewLink(field.text.toString(), false) }
            .setNegativeButton(R.string.cancel, null)
            .create()
        pasteButton.setOnClickListener {
            val clip = (getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager).primaryClip?.getItemAt(0)?.coerceToText(this)?.toString().orEmpty()
            dialog.dismiss()
            reviewLink(clip, true)
        }
        scanButton.setOnClickListener {
            dialog.dismiss()
            scanner.launch(Intent(this, ScannerActivity::class.java))
        }
        dialog.show()
    }

    private fun reviewLink(raw: String, clearClipboard: Boolean) {
        val invite = try { InviteValidator.parseMobileLink(raw) } catch (_: Exception) { message(getString(R.string.pairing_rejected_title), getString(R.string.pairing_rejected_body)); return }
        val hosts = invite.signalingUrls.mapNotNull { runCatching { java.net.URI(it).authority }.getOrNull() }.joinToString(", ")
        val expiry = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm z").withZone(ZoneId.systemDefault()).format(invite.expiresAt)
        AlertDialog.Builder(this).setTitle(getString(R.string.pair_confirm_title, invite.displayName)).setMessage(getString(R.string.pair_confirm_body, expiry, hosts))
            .setPositiveButton(R.string.pair_securely) { _, _ ->
                try {
                    repository.pair(invite)
                    if (clearClipboard) clearClipboard()
                    showInstances()
                } catch (_: Exception) { message(getString(R.string.pairing_failed_title), getString(R.string.pairing_failed_body)) }
            }.setNegativeButton(R.string.cancel, null).show()
    }

    private fun clearClipboard() {
        val manager = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        if (Build.VERSION.SDK_INT >= 28) manager.clearPrimaryClip() else manager.setPrimaryClip(ClipData.newPlainText("", ""))
    }

    private fun confirmForget(instance: MobileInstance) {
        AlertDialog.Builder(this).setTitle(getString(R.string.forget_confirm_title, instance.displayName)).setMessage(R.string.forget_confirm_body)
            .setPositiveButton(R.string.forget) { _, _ ->
                val removed = runCatching { repository.forget(instance.id) { check(profileStore.deleteProfile(it)) } }
                showInstances()
                if (removed.isFailure) message(getString(R.string.profile_cleanup_title), getString(R.string.profile_cleanup_body))
            }
            .setNegativeButton(R.string.cancel, null).show()
    }

    private fun openInstance(instance: MobileInstance) {
        val password = secretStore.get(instance.passwordRef); val capability = secretStore.get(instance.capabilityRef)
        if (password.isNullOrEmpty() || capability.isNullOrEmpty()) { message(getString(R.string.pairing_unavailable_title), getString(R.string.pairing_unavailable_body)); return }
        root.removeAllViews()
        val sourceRevision = runCatching { assets.open("base-manifest.json").use { Json.parseToJsonElement(it.readBytes().decodeToString()).jsonObject["source_revision"]!!.jsonPrimitive.content } }.getOrDefault("unknown")
        lateinit var office: OfficePackCoordinator
        val surface = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            background = ContextCompat.getDrawable(context, R.drawable.office_surface_bg)
            setPadding(dp(16), dp(10), dp(16), dp(10))
        }
        val status = TextView(this).apply { setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f); maxLines = 4 }
        val progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 100 }
        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val download = Button(this).apply { setText(R.string.download); minHeight = dp(48); setOnClickListener { office.download() } }
        val cancel = Button(this).apply { setText(R.string.cancel); minHeight = dp(48); setOnClickListener { office.cancel() } }
        actions.addView(download, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(6); topMargin = dp(8) })
        actions.addView(cancel, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(6); topMargin = dp(8) })
        surface.addView(status, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        surface.addView(progress, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(8) })
        surface.addView(actions, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        office = OfficePackCoordinator(this, sourceRevision, BuildConfig.VERSION_NAME) { state -> runOnUiThread {
            surface.visibility = if (state is OfficeState.Idle || state is OfficeState.Active) View.GONE else View.VISIBLE
            when (state) {
                is OfficeState.AwaitingConsent -> { status.text = getString(R.string.office_consent, android.text.format.Formatter.formatFileSize(this, state.totalBytes)); progress.progress = 0; download.isEnabled = true }
                is OfficeState.Downloading -> { status.setText(R.string.office_downloading); progress.progress = (state.progress * 100).toInt(); download.isEnabled = false }
                is OfficeState.Canceled -> { status.setText(R.string.office_canceled); download.isEnabled = true }
                is OfficeState.Offline -> { status.setText(R.string.office_offline); download.isEnabled = true }
                is OfficeState.Failed -> { status.text = state.message; download.isEnabled = true }
                else -> Unit
            }
        } }
        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            minimumHeight = dp(56)
            setPadding(dp(4), 0, dp(16), 0)
        }
        topBar.addView(Button(this).apply {
            setText(R.string.back_to_instances)
            minHeight = dp(48)
            setOnClickListener { showInstances() }
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        topBar.addView(TextView(this).apply {
            text = instance.displayName
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
            setTypeface(typeface, Typeface.BOLD)
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
            setPadding(dp(12), 0, 0, 0)
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        val host = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        host.addView(topBar, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        val hosted = try { createBusinessOsWebView(this, instance, password, capability, profileStore, office) } catch (_: Exception) { message(getString(R.string.launch_failed_title), getString(R.string.launch_failed_body)); showInstances(); return }
        webView = hosted
        host.addView(hosted, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        host.addView(surface, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(host, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
    }

    private fun message(title: String, body: String) { AlertDialog.Builder(this).setTitle(title).setMessage(body).setPositiveButton(android.R.string.ok, null).show() }
}
