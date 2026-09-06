package com.kanjilens.translate

import android.graphics.Bitmap
import android.util.Base64
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslatorOptions
import com.kanjilens.data.models.AppSettings
import com.kanjilens.data.models.TranslationPair
import com.kanjilens.ocr.RecognizedTextBlock
import com.kanjilens.ocr.TextRecognizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.UnknownHostException
import java.util.concurrent.TimeUnit

sealed class TranslateResult {
    /** Offline translation may also include sentence-aware source/target pairs. */
    data class Success(
        val text: String,
        val original: String? = null,
        val translationBlocks: List<TranslationPair>? = null,
    ) : TranslateResult()
    data class Error(val message: String) : TranslateResult()
}

class ScreenTranslator(
    private val textRecognizer: TextRecognizer,
) {

    companion object {
        const val STYLE_AUTO = 0
        const val STYLE_TRANSLATE_ONLY = 1
        const val STYLE_TRANSLATE_AND_EXPLAIN = 2

        /** Upper bound on characters handed to ML Kit in a single translate() call. */
        private const val MAX_TRANSLATE_CHARS = 2000
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    private var mlKitTranslator: com.google.mlkit.nl.translate.Translator? = null
    private var mlKitCurrentLangPair: String? = null

    private fun langPair(source: String, target: String): String =
        "${mlKitLanguageCode(source)}->${mlKitLanguageCode(target)}"

    private fun mlKitLanguageCode(appLangCode: String): String {
        return when (appLangCode) {
            AppSettings.LANG_JAPANESE -> TranslateLanguage.JAPANESE
            AppSettings.LANG_ENGLISH -> TranslateLanguage.ENGLISH
            AppSettings.LANG_SPANISH -> TranslateLanguage.SPANISH
            AppSettings.LANG_PORTUGUESE -> TranslateLanguage.PORTUGUESE
            AppSettings.LANG_FRENCH -> TranslateLanguage.FRENCH
            AppSettings.LANG_GERMAN -> TranslateLanguage.GERMAN
            AppSettings.LANG_ITALIAN -> TranslateLanguage.ITALIAN
            AppSettings.LANG_CHINESE -> TranslateLanguage.CHINESE
            AppSettings.LANG_KOREAN -> TranslateLanguage.KOREAN
            AppSettings.LANG_RUSSIAN -> TranslateLanguage.RUSSIAN
            else -> TranslateLanguage.ENGLISH
        }
    }

    suspend fun ensureOfflineModelReady(
        sourceLanguage: String = AppSettings.LANG_JAPANESE,
        targetLang: String = AppSettings.LANG_ENGLISH,
    ) {
        val pair = langPair(sourceLanguage, targetLang)
        if (mlKitTranslator != null && mlKitCurrentLangPair == pair) return
        withContext(Dispatchers.IO) {
            mlKitTranslator?.close()
            val options = TranslatorOptions.Builder()
                .setSourceLanguage(mlKitLanguageCode(sourceLanguage))
                .setTargetLanguage(mlKitLanguageCode(targetLang))
                .build()
            val translator = Translation.getClient(options)
            val conditions = DownloadConditions.Builder().build()
            translator.downloadModelIfNeeded(conditions).await()
            mlKitTranslator = translator
            mlKitCurrentLangPair = pair
        }
    }

    suspend fun translateScreen(
        bitmap: Bitmap,
        apiKey: String,
        style: Int = STYLE_AUTO,
        model: Int = AppSettings.MODEL_GPT4O_MINI,
        outputLanguage: String = AppSettings.LANG_ENGLISH,
        sourceLanguage: String = AppSettings.LANG_JAPANESE,
        ollamaUrl: String = "",
        ollamaModel: String = "",
        ollamaVision: Boolean = true,
        customUrl: String = "",
        customApiKey: String = "",
        customModel: String = "",
        customVision: Boolean = true,
        onDownloading: (() -> Unit)? = null,
    ): TranslateResult {
        return withContext(Dispatchers.IO) {
            try {
                if (model == AppSettings.MODEL_MLKIT_OFFLINE || model == AppSettings.MODEL_MLKIT_OFFLINE_AUTO) {
                    return@withContext translateOffline(bitmap, sourceLanguage, outputLanguage, onDownloading)
                }

                if (model == AppSettings.MODEL_OLLAMA || model == AppSettings.MODEL_CUSTOM) {
                    val endpoint = if (model == AppSettings.MODEL_OLLAMA) {
                        "${ollamaUrl.trimEnd('/')}/v1/chat/completions"
                    } else {
                        "${customUrl.trimEnd('/')}/v1/chat/completions"
                    }
                    val key = if (model == AppSettings.MODEL_CUSTOM) customApiKey else ""
                    val modelName = if (model == AppSettings.MODEL_OLLAMA) ollamaModel else customModel
                    val vision = if (model == AppSettings.MODEL_OLLAMA) ollamaVision else customVision
                    val prompt = getSystemPrompt(style, outputLanguage)

                    val base64 = if (vision) bitmapToBase64(bitmap) else ""
                    val preferredScript = AppSettings.ocrScriptFor(sourceLanguage)
                    textRecognizer.setScript(preferredScript)
                    val sourceGroups = if (vision) {
                        textRecognizer.recognizeStructuredTextBlocks(bitmap)
                    } else {
                        // Text-only Ollama models need OCR. Try all bundled OCR
                        // scripts so an incorrect source-language setting does
                        // not turn a readable screen into an empty request.
                        textRecognizer.recognizeStructuredTextBlocksAnyScript(bitmap, preferredScript)
                    }
                        ?.let { groupForTranslation(it) }
                        ?.takeIf { it.isNotEmpty() }
                    val sourceText = sourceGroups?.joinToString("\n")

                    if (sourceGroups == null) {
                        val fallbackOcrText = if (!vision) {
                            textRecognizer.recognizeStructuredTextBlocksAnyScript(bitmap, preferredScript)
                                ?.joinToString("\n") { it.text }
                        } else {
                            null
                        }
                        val result = callOpenAICompatible(
                            base64Image = base64,
                            endpoint = endpoint,
                            apiKey = key,
                            model = modelName,
                            vision = vision,
                            systemPrompt = prompt,
                            ocrText = fallbackOcrText,
                            focusText = fallbackOcrText,
                        )
                        return@withContext if (result != null) {
                            TranslateResult.Success(text = result)
                        } else if (!vision && fallbackOcrText == null) {
                            TranslateResult.Error("No text found. Select the correct source language or enable vision mode.")
                        } else {
                            TranslateResult.Error(compatibleModelError(model))
                        }
                    }

                    val pairs = mutableListOf<TranslationPair>()
                    for (group in sourceGroups) {
                        var translated = callOpenAICompatible(
                            base64Image = base64,
                            endpoint = endpoint,
                            apiKey = key,
                            model = modelName,
                            vision = vision,
                            systemPrompt = prompt,
                            ocrText = group,
                            focusText = group,
                        )

                        // Ollama's model list exposes text-only models as well as
                        // vision models. If the default vision request is rejected,
                        // retry the same OCR block as a text request.
                        if (translated == null && vision) {
                            translated = callOpenAICompatible(
                                base64Image = "",
                                endpoint = endpoint,
                                apiKey = key,
                                model = modelName,
                                vision = false,
                                systemPrompt = prompt,
                                ocrText = group,
                                focusText = group,
                            )
                        }

                        if (translated == null) {
                            return@withContext TranslateResult.Error(compatibleModelError(model))
                        }
                        pairs += TranslationPair(original = group, translation = translated)
                    }

                    return@withContext TranslateResult.Success(
                        text = pairs.joinToString("\n") { it.translation },
                        original = sourceText,
                        translationBlocks = pairs,
                    )
                }

                val base64Image = bitmapToBase64(bitmap)
                val prompt = getSystemPrompt(style, outputLanguage)

                val result = when (model) {
                    AppSettings.MODEL_GEMINI_FLASH -> callGemini(base64Image, apiKey, prompt)
                    else -> callOpenAI(base64Image, apiKey, prompt)
                }

                if (result != null) {
                    TranslateResult.Success(result)
                } else {
                    TranslateResult.Error(
                        "Cloud translation failed (model ID: $model). Check the selected provider and API key."
                    )
                }
            } catch (e: UnknownHostException) {
                TranslateResult.Error("No internet connection")
            } catch (e: java.net.SocketTimeoutException) {
                TranslateResult.Error("Connection timed out. Try again.")
            } catch (e: Exception) {
                e.printStackTrace()
                TranslateResult.Error("Translation failed: ${e.message ?: "unknown error"}")
            }
        }
    }

    private suspend fun translateOffline(
        bitmap: Bitmap,
        sourceLanguage: String = AppSettings.LANG_JAPANESE,
        outputLanguage: String = AppSettings.LANG_ENGLISH,
        onDownloading: (() -> Unit)? = null,
    ): TranslateResult {
        textRecognizer.setScript(AppSettings.ocrScriptFor(sourceLanguage))

        val recognizedBlocks = textRecognizer.recognizeStructuredTextBlocks(bitmap)
            ?: return TranslateResult.Error("No text found in screenshot")

        if (recognizedBlocks.isEmpty()) {
            return TranslateResult.Error("No text found in screenshot")
        }

        val needsDownload =
            mlKitTranslator == null || mlKitCurrentLangPair != langPair(sourceLanguage, outputLanguage)
        if (needsDownload) {
            withContext(Dispatchers.Main) { onDownloading?.invoke() }
        }

        try {
            ensureOfflineModelReady(sourceLanguage, outputLanguage)
        } catch (e: Exception) {
            return TranslateResult.Error("Download the offline model first. Connect to WiFi and try again.")
        }

        val translator = mlKitTranslator
            ?: return TranslateResult.Error("Offline translator not available")

        val translationGroups = groupForTranslation(recognizedBlocks)
        val originalText = translationGroups.joinToString("\n")

        return try {
            val pairs = translationGroups.map { group ->
                TranslationPair(
                    original = group,
                    translation = translateGroupPreservingLineBreaks(group, translator),
                )
            }
            val translated = pairs.joinToString("\n") { it.translation }

            TranslateResult.Success(
                text = translated.trim(),
                original = originalText,
                translationBlocks = pairs,
            )
        } catch (e: Exception) {
            TranslateResult.Error("Offline translation failed: ${e.message ?: "unknown error"}")
        }
    }

    private suspend fun translateGroupPreservingLineBreaks(
        group: String,
        translator: com.google.mlkit.nl.translate.Translator,
    ): String {
        if (!group.contains('\n')) return translator.translate(group).await()

        val lineBreakMarker = '\uE000'
        val expectedBreaks = group.count { it == '\n' }
        val markedGroup = group.replace("\n", " $lineBreakMarker ")
        val markedTranslation = translator.translate(markedGroup).await()
        val translatedBreaks = markedTranslation.count { it == lineBreakMarker }

        if (translatedBreaks >= expectedBreaks) {
            return markedTranslation
                .replace(lineBreakMarker.toString(), "\n")
                .replace(Regex("[ \\t]*\\n[ \\t]*"), "\n")
        }

        // Some downloaded ML Kit models discard unknown marker characters. The
        // fallback keeps the visual rows correct when that happens.
        val translatedLines = StringBuilder()
        for (line in group.split('\n')) {
            if (translatedLines.isNotEmpty()) translatedLines.append('\n')
            if (line.isNotBlank()) translatedLines.append(translator.translate(line).await())
        }
        return translatedLines.toString()
    }

    /**
     * Joins OCR blocks that are likely parts of the same sentence. A block is
     * kept separate when the previous one ends punctuation, is far away on the
     * screen, or the group would exceed ML Kit's input limit.
     */
    private fun groupForTranslation(blocks: List<RecognizedTextBlock>): List<String> {
        val ordered = blocks.sortedWith(
            compareBy<RecognizedTextBlock> { it.boundingBox?.top ?: Int.MAX_VALUE }
                .thenBy { it.boundingBox?.left ?: Int.MAX_VALUE },
        )
        val groups = mutableListOf<String>()
        val current = StringBuilder()
        var previous: RecognizedTextBlock? = null

        fun flush() {
            if (current.isNotEmpty()) {
                groups.add(current.toString())
                current.clear()
            }
        }

        for (block in ordered) {
            val canJoin = previous != null &&
                !endsSentence(previous.text) &&
                current.length + block.text.length + 1 <= MAX_TRANSLATE_CHARS &&
                areContinuous(previous, block)

            if (!canJoin) flush()
            if (current.isNotEmpty()) {
                current.append(
                    if (preserveSemanticLineBreak(previous, block)) '\n' else ' ',
                )
            }
            current.append(block.text)
            previous = block

            if (endsSentence(block.text)) flush()
        }
        flush()
        return groups
    }

    private fun endsSentence(text: String): Boolean {
        val last = text.trimEnd().lastOrNull() ?: return false
        return last in "。！？!?…;；" || last == '.'
    }

    /**
     * Keeps a line break for speaker/name labels, but turns OCR's visual wraps
     * inside a sentence into spaces. For example, `Roland` stays above the
     * dialogue, while `... in the` + `medical arts.` becomes one sentence.
     */
    private fun preserveSemanticLineBreak(
        previous: RecognizedTextBlock?,
        current: RecognizedTextBlock,
    ): Boolean {
        val previousText = previous?.text?.trim().orEmpty()
        if (previousText.isEmpty()) return false

        val isShortSingleLineLabel = previousText.length <= 32 &&
            previousText.none { it.isWhitespace() } &&
            !endsSentence(previousText)
        if (isShortSingleLineLabel) return true

        val previousBox = previous?.boundingBox ?: return false
        val currentBox = current.boundingBox ?: return false
        val lineHeight = maxOf(previousBox.height(), currentBox.height(), 1)
        val verticalGap = currentBox.top - previousBox.bottom
        return verticalGap > lineHeight
    }

    private fun areContinuous(previous: RecognizedTextBlock?, current: RecognizedTextBlock): Boolean {
        val previousBox = previous?.boundingBox ?: return true
        val currentBox = current.boundingBox ?: return true
        val lineHeight = maxOf(previousBox.height(), currentBox.height(), 1)
        val verticalGap = currentBox.top - previousBox.bottom
        val horizontalShift = kotlin.math.abs(currentBox.left - previousBox.left)

        // Separate columns or distant UI regions should not be merged even if
        // OCR omitted punctuation from both labels.
        return verticalGap <= lineHeight * 3 &&
            horizontalShift <= maxOf(previousBox.width(), currentBox.width()) * 1.5
    }

    private fun callOpenAI(base64Image: String, apiKey: String, systemPrompt: String): String? {
        val body = JSONObject().apply {
            put("model", "gpt-4o-mini")
            put("max_tokens", 1000)
            put("messages", JSONArray().apply {
                put(JSONObject().apply {
                    put("role", "system")
                    put("content", systemPrompt)
                })
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", JSONArray().apply {
                        put(JSONObject().apply {
                            put("type", "text")
                            put("text", "Translate this game screen.")
                        })
                        put(JSONObject().apply {
                            put("type", "image_url")
                            put("image_url", JSONObject().apply {
                                put("url", "data:image/jpeg;base64,$base64Image")
                                put("detail", "low")
                            })
                        })
                    })
                })
            })
        }

        val request = Request.Builder()
            .url("https://api.openai.com/v1/chat/completions")
            .addHeader("Authorization", "Bearer $apiKey")
            .addHeader("Content-Type", "application/json")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = client.newCall(request).execute()
        val responseBody = response.body?.string() ?: return null
        if (!response.isSuccessful) return null

        val json = JSONObject(responseBody)
        val choices = json.getJSONArray("choices")
        return if (choices.length() > 0) {
            choices.getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
        } else null
    }

    private fun callGemini(base64Image: String, apiKey: String, systemPrompt: String): String? {
        val body = JSONObject().apply {
            put("system_instruction", JSONObject().apply {
                put("parts", JSONArray().apply {
                    put(JSONObject().apply { put("text", systemPrompt) })
                })
            })
            put("contents", JSONArray().apply {
                put(JSONObject().apply {
                    put("parts", JSONArray().apply {
                        put(JSONObject().apply {
                            put("text", "Translate this game screen.")
                        })
                        put(JSONObject().apply {
                            put("inline_data", JSONObject().apply {
                                put("mime_type", "image/jpeg")
                                put("data", base64Image)
                            })
                        })
                    })
                })
            })
            put("generationConfig", JSONObject().apply {
                put("maxOutputTokens", 1000)
            })
        }

        val request = Request.Builder()
            .url("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$apiKey")
            .addHeader("Content-Type", "application/json")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = client.newCall(request).execute()
        val responseBody = response.body?.string() ?: return null
        if (!response.isSuccessful) return null

        val json = JSONObject(responseBody)
        val candidates = json.optJSONArray("candidates") ?: return null
        if (candidates.length() == 0) return null

        return candidates.getJSONObject(0)
            .getJSONObject("content")
            .getJSONArray("parts")
            .getJSONObject(0)
            .getString("text")
    }

    private fun compatibleModelError(model: Int): String = if (model == AppSettings.MODEL_OLLAMA) {
        "Ollama request failed. Check the server address and model name. API key is not required."
    } else {
        "Translation failed. Check your endpoint and model settings."
    }

    private fun callOpenAICompatible(
        base64Image: String,
        endpoint: String,
        apiKey: String,
        model: String,
        vision: Boolean,
        systemPrompt: String,
        ocrText: String? = null,
        focusText: String? = null,
    ): String? {
        val userContent = if (vision) {
            JSONArray().apply {
                put(JSONObject().apply {
                    put("type", "text")
                    put(
                        "text",
                        if (focusText.isNullOrBlank()) {
                            "Translate this game screen."
                        } else {
                            "Translate only this text block from the screen. Do not translate other visible text:\n\n$focusText"
                        },
                    )
                })
                put(JSONObject().apply {
                    put("type", "image_url")
                    put("image_url", JSONObject().apply {
                        put("url", "data:image/jpeg;base64,$base64Image")
                        put("detail", "low")
                    })
                })
            }
        } else {
            focusText ?: ocrText ?: return null
        }

        val body = JSONObject().apply {
            put("model", model)
            put("max_tokens", 1000)
            put("messages", JSONArray().apply {
                put(JSONObject().apply {
                    put("role", "system")
                    put("content", systemPrompt)
                })
                put(JSONObject().apply {
                    put("role", "user")
                    if (vision) {
                        put("content", userContent)
                    } else {
                        put("content", "Translate this game screen text:\n\n$userContent")
                    }
                })
            })
        }

        val requestBuilder = Request.Builder()
            .url(endpoint)
            .addHeader("Content-Type", "application/json")
            .post(body.toString().toRequestBody("application/json".toMediaType()))

        if (apiKey.isNotEmpty()) {
            requestBuilder.addHeader("Authorization", "Bearer $apiKey")
        }

        val response = client.newCall(requestBuilder.build()).execute()
        val responseBody = response.body?.string() ?: return null
        if (!response.isSuccessful) return null

        val json = JSONObject(responseBody)
        val choices = json.optJSONArray("choices") ?: return null
        return if (choices.length() > 0) {
            choices.getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
        } else null
    }

    suspend fun fetchOllamaModels(baseUrl: String): List<String> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/tags")
                .build()
            val response = client.newCall(request).execute()
            if (!response.isSuccessful) return@withContext emptyList()
            val body = response.body?.string() ?: return@withContext emptyList()
            val json = JSONObject(body)
            val models = json.optJSONArray("models") ?: return@withContext emptyList()
            (0 until models.length()).map { models.getJSONObject(it).getString("name") }
        } catch (e: Exception) {
            emptyList()
        }
    }

    suspend fun fetchOpenAIModels(baseUrl: String, apiKey: String): List<String> = withContext(Dispatchers.IO) {
        try {
            val requestBuilder = Request.Builder().url("$baseUrl/v1/models")
            if (apiKey.isNotEmpty()) requestBuilder.addHeader("Authorization", "Bearer $apiKey")
            val response = client.newCall(requestBuilder.build()).execute()
            if (!response.isSuccessful) return@withContext emptyList()
            val body = response.body?.string() ?: return@withContext emptyList()
            val json = JSONObject(body)
            val data = json.optJSONArray("data") ?: return@withContext emptyList()
            (0 until data.length()).map { data.getJSONObject(it).getString("id") }
        } catch (e: Exception) {
            emptyList()
        }
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val stream = ByteArrayOutputStream()
        val scaled = if (bitmap.width > 1024) {
            val ratio = 1024f / bitmap.width
            Bitmap.createScaledBitmap(
                bitmap,
                1024,
                (bitmap.height * ratio).toInt(),
                true,
            )
        } else {
            bitmap
        }
        scaled.compress(Bitmap.CompressFormat.JPEG, 80, stream)
        return Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)
    }

    private fun getSystemPrompt(style: Int, outputLanguage: String = AppSettings.LANG_ENGLISH): String {
        val langName = AppSettings.languageDisplayName(outputLanguage)
        val baseRules = "Never be conversational. No greetings, no questions, no \"feel free to ask\", no \"let me know\". No markdown formatting."

        return when (style) {
            STYLE_TRANSLATE_ONLY -> {
                "You translate game screenshots to $langName. The text may be in any language (Japanese, Chinese, Korean, etc). " +
                    "Translate all visible text on screen. " +
                    "For menus, list each option translated. " +
                    "For dialogue, translate naturally. " +
                    "For stats, translate the labels and values. " +
                    "Only translate, do not explain or give advice. " +
                    "Always respond in $langName. " +
                    baseRules
            }
            STYLE_TRANSLATE_AND_EXPLAIN -> {
                "You are a game assistant helping someone play a game that's not in their language. The screen may be in any language (Japanese, Chinese, Korean, etc).\n\n" +
                    "Rules:\n" +
                    "- First: translate all text on screen to $langName\n" +
                    "- Then: explain what you're looking at and what you should do to progress\n" +
                    "- For menus: translate each option and recommend which to pick\n" +
                    "- For dialogue/story: translate naturally, then summarize what's happening\n" +
                    "- For gameplay/instructions: translate and explain what the game wants you to do\n" +
                    "- For stats/progress: explain the key numbers and what they mean\n" +
                    "- Talk directly to the user using \"you\" (e.g. \"you need to select...\", \"your stats are...\")\n" +
                    "- Keep it concise but useful\n" +
                    "- Always respond in $langName\n" +
                    "- $baseRules"
            }
            else -> { // AUTO
                "You are a game assistant helping someone play a game that's not in their language. The screen may be in any language (Japanese, Chinese, Korean, etc).\n\n" +
                    "Always do both:\n" +
                    "1. Translate all text on screen to $langName\n" +
                    "2. Briefly explain what you're seeing and what to do next\n\n" +
                    "- For menus: translate each option and say which one to pick to progress\n" +
                    "- For dialogue/story: translate naturally, then summarize what's happening\n" +
                    "- For gameplay/instructions: translate and explain what the game wants you to do\n" +
                    "- For stats/progress: explain the key numbers and what they mean\n" +
                    "- Talk directly to the user using \"you\" (e.g. \"you need to select...\", \"your health is...\")\n" +
                    "- Keep it concise — you just want to keep playing\n" +
                    "- Always respond in $langName\n" +
                    "- $baseRules"
            }
        }
    }
}
