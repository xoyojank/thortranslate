package com.kanjilens.translate

import android.content.Context
import android.os.StatFs
import android.os.SystemClock
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlin.coroutines.coroutineContext

fun interface HyMt2TokenListener {
    fun onToken(token: String)
}

sealed class HyMt2DownloadState {
    data object NotInstalled : HyMt2DownloadState()
    data class Downloading(val downloadedBytes: Long, val totalBytes: Long?) : HyMt2DownloadState()
    data object Ready : HyMt2DownloadState()
    data class Error(val message: String) : HyMt2DownloadState()
}

/**
 * Experimental on-device runner for a separately downloaded Hy-MT2-1.8B Q4 GGUF.
 * The model is not bundled in the APK and is stored in app-specific external storage.
 */
class HyMt2Engine(context: Context) : AutoCloseable {

    companion object {
        private const val MODEL_DIRECTORY = "hymt2"
        private const val MODEL_FILE_NAME = "Hy-MT2-1.8B-Q4_K_M.gguf"
        private const val MODEL_DOWNLOAD_URL =
            "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/main/Hy-MT2-1.8B-Q4_K_M.gguf"
        private const val MIN_MODEL_BYTES = 1_000_000_000L
        private const val REQUIRED_FREE_BYTES = 2_500_000_000L
        // Tuned for a Snapdragon 8 Gen 2-class device and short game-dialogue translation.
        private const val CONTEXT_SIZE = 1024
        private const val DEFAULT_THREAD_COUNT = 6
        private const val MAX_OUTPUT_TOKENS = 64

        init {
            System.loadLibrary("hymt2_jni")
        }
    }

    private val modelDirectory: File =
        context.getExternalFilesDir(MODEL_DIRECTORY) ?: File(context.filesDir, MODEL_DIRECTORY)
    private val modelFile = File(modelDirectory, MODEL_FILE_NAME)
    private val partialFile = File(modelDirectory, "$MODEL_FILE_NAME.part")
    private val handle = AtomicLong(0L)
    private var activeThreadCount = 0
    private val lock = Any()
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.SECONDS)
        .build()

    private val _downloadState = MutableStateFlow(initialDownloadState())
    val downloadState: StateFlow<HyMt2DownloadState> = _downloadState
    private val _lastInferenceMillis = MutableStateFlow<Long?>(null)
    val lastInferenceMillis: StateFlow<Long?> = _lastInferenceMillis

    val modelPath: File
        get() = modelFile

    val isModelAvailable: Boolean
        get() = modelFile.isFile && modelFile.length() >= MIN_MODEL_BYTES

    suspend fun downloadModel() = withContext(Dispatchers.IO) {
        if (isModelAvailable) {
            _downloadState.value = HyMt2DownloadState.Ready
            return@withContext
        }

        modelDirectory.mkdirs()
        val availableBytes = StatFs(modelDirectory.absolutePath).availableBytes
        if (availableBytes < REQUIRED_FREE_BYTES) {
            _downloadState.value = HyMt2DownloadState.Error(
                "Not enough free storage. Hy-MT2 needs at least 2.5 GB available."
            )
            return@withContext
        }

        partialFile.delete()
        try {
            val request = Request.Builder()
                .url(MODEL_DOWNLOAD_URL)
                .header("User-Agent", "ThorLens/1.0")
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    throw IllegalStateException("Hugging Face returned HTTP ${response.code}")
                }
                val body = response.body ?: throw IllegalStateException("Empty model download response")
                val totalBytes = body.contentLength().takeIf { it > 0L }
                if (totalBytes != null && totalBytes < MIN_MODEL_BYTES) {
                    throw IllegalStateException("Downloaded file is too small to be the Hy-MT2 Q4 model")
                }

                body.byteStream().use { input ->
                    FileOutputStream(partialFile).use { output ->
                        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                        var downloadedBytes = 0L
                        while (true) {
                            coroutineContext.ensureActive()
                            val count = input.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                            downloadedBytes += count
                            _downloadState.value = HyMt2DownloadState.Downloading(downloadedBytes, totalBytes)
                        }
                        output.fd.sync()
                    }
                }
            }

            if (partialFile.length() < MIN_MODEL_BYTES) {
                throw IllegalStateException("The downloaded model is incomplete. Try again on a stable Wi-Fi connection.")
            }
            if (!partialFile.renameTo(modelFile)) {
                throw IllegalStateException("Unable to finalize the downloaded model file.")
            }
            _downloadState.value = HyMt2DownloadState.Ready
        } catch (error: Exception) {
            partialFile.delete()
            _downloadState.value = HyMt2DownloadState.Error(
                error.message ?: "Hy-MT2 model download failed."
            )
        }
    }

    fun deleteModel() {
        close()
        partialFile.delete()
        modelFile.delete()
        _downloadState.value = HyMt2DownloadState.NotInstalled
    }

    suspend fun translate(
        sourceText: String,
        targetLanguage: String,
        threads: Int = DEFAULT_THREAD_COUNT,
        onPartial: (String) -> Unit = {},
    ): String = withContext(Dispatchers.Default) {
        require(sourceText.isNotBlank()) { "No text to translate." }
        require(threads in setOf(4, 6, 8)) { "Hy-MT2 threads must be 4, 6, or 8" }
        require(isModelAvailable) {
            "Hy-MT2 model not found. Download the model in Settings first."
        }

        synchronized(lock) {
            if (activeThreadCount != 0 && activeThreadCount != threads) releaseLocked()
            val nativeHandle = handle.updateAndGet { current ->
                if (current != 0L) current
                else nativeCreate(modelFile.absolutePath, CONTEXT_SIZE, threads)
            }
            activeThreadCount = threads

            val startedAt = SystemClock.elapsedRealtime()
            val result = sanitizeTranslation(nativeTranslate(
                nativeHandle,
                buildPrompt(sourceText, targetLanguage),
                MAX_OUTPUT_TOKENS,
                HyMt2TokenListener(onPartial),
            ))
            _lastInferenceMillis.value = SystemClock.elapsedRealtime() - startedAt
            result
        }
    }

    private fun initialDownloadState(): HyMt2DownloadState =
        if (isModelAvailable) HyMt2DownloadState.Ready else HyMt2DownloadState.NotInstalled

    private fun sanitizeTranslation(raw: String): String {
        val terminators = listOf(
            "<|im_end|>", "<|im_start|>", "<source>", "</source>",
            "<code>", "</code>", "<font", "</font>",
        )
        val end = terminators
            .map { raw.indexOf(it) }
            .filter { it >= 0 }
            .minOrNull() ?: raw.length
        return raw.substring(0, end).trim()
    }

    private fun buildPrompt(sourceText: String, targetLanguage: String): String =
        "Translate the text between <source> and </source> into $targetLanguage. " +
            "Only output the translation. Do not add dialogue, explanations, apologies, speaker names, or any text not present in the source. " +
            "Keep proper names unchanged and preserve every source line break. " +
            "<source>\n$sourceText\n</source>"

    override fun close() {
        synchronized(lock) { releaseLocked() }
    }

    private fun releaseLocked() {
        handle.getAndSet(0L).takeIf { it != 0L }?.let(::nativeDestroy)
        activeThreadCount = 0
    }

    private external fun nativeCreate(modelPath: String, contextSize: Int, threads: Int): Long
    private external fun nativeTranslate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        listener: HyMt2TokenListener,
    ): String
    private external fun nativeDestroy(handle: Long)
}
