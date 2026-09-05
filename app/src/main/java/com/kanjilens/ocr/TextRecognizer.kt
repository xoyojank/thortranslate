package com.kanjilens.ocr

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer as MlKitTextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.kanjilens.data.models.AppSettings
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/**
 * On-device OCR. ML Kit ships a separate recogniser per script, so the caller
 * picks one with [setScript] before recognising. Defaults to Japanese to keep
 * JP Dictionary mode working without any setup.
 */
class TextRecognizer {

    companion object {
        private const val TAG = "KanjiLens"

        private fun newRecognizer(script: String): MlKitTextRecognizer = when (script) {
            AppSettings.SCRIPT_CHINESE ->
                TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build())
            AppSettings.SCRIPT_KOREAN ->
                TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
            AppSettings.SCRIPT_LATIN ->
                TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            else ->
                TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build())
        }
    }

    private val lock = Any()

    @Volatile
    private var currentScript: String = AppSettings.SCRIPT_JAPANESE

    @Volatile
    private var recognizer: MlKitTextRecognizer = newRecognizer(AppSettings.SCRIPT_JAPANESE)

    /** Swaps in the recogniser for [script]. No-op if it is already active. */
    fun setScript(script: String) {
        synchronized(lock) {
            if (script == currentScript) return
            val previous = recognizer
            currentScript = script
            recognizer = newRecognizer(script)
            previous.close()
        }
    }

    private fun activeRecognizer(): MlKitTextRecognizer = synchronized(lock) { recognizer }

    suspend fun recognizeText(bitmap: Bitmap): String? = suspendCancellableCoroutine { continuation ->
        val image = InputImage.fromBitmap(bitmap, 0)

        Log.d(TAG, "OCR: Starting text recognition on ${bitmap.width}x${bitmap.height} image")

        activeRecognizer().process(image)
            .addOnSuccessListener { result ->
                val text = result.text.trim()
                Log.d(TAG, "OCR: Recognized ${result.textBlocks.size} blocks, text length=${text.length}")
                if (text.isNotEmpty()) {
                    Log.d(TAG, "OCR: Text = $text")
                    continuation.resume(text)
                } else {
                    Log.d(TAG, "OCR: No text found")
                    continuation.resume(null)
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "OCR: Recognition failed", e)
                continuation.resume(null)
            }
    }

    suspend fun recognizeTextBlocks(bitmap: Bitmap): List<String>? = suspendCancellableCoroutine { continuation ->
        val image = InputImage.fromBitmap(bitmap, 0)

        activeRecognizer().process(image)
            .addOnSuccessListener { result ->
                val blocks = result.textBlocks
                    .map { it.text.trim() }
                    .filter { it.isNotEmpty() }
                if (blocks.isNotEmpty()) {
                    continuation.resume(blocks)
                } else {
                    continuation.resume(null)
                }
            }
            .addOnFailureListener {
                continuation.resume(null)
            }
    }

    fun close() {
        activeRecognizer().close()
    }
}
