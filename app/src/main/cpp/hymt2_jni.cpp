#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

namespace {
constexpr const char * kTag = "HyMt2Engine";

struct Engine {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_sampler * sampler = nullptr;
    const llama_vocab * vocab = nullptr;
    int context_size = 2048;
    std::mutex mutex;

    ~Engine() {
        if (sampler != nullptr) llama_sampler_free(sampler);
        if (context != nullptr) llama_free(context);
        if (model != nullptr) llama_model_free(model);
    }
};

std::string jstring_to_string(JNIEnv * env, jstring value) {
    if (value == nullptr) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

void throw_error(JNIEnv * env, const std::string & message) {
    jclass exception = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(exception, message.c_str());
}

std::string format_hymt2_chat_prompt(const std::string & instruction) {
    // Hy-MT2's GGUF is chat-tuned. The official llama.cpp invocation uses
    // --jinja; feeding the translation instruction as raw completion text
    // makes the model continue unrelated dialogue training examples instead.
    return "<|im_start|>user\n" + instruction +
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & prompt) {
    const int count = -llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (count <= 0) return {};

    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(), count, true, true) < 0) {
        return {};
    }
    return tokens;
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_kanjilens_translate_HyMt2Engine_nativeCreate(
    JNIEnv * env,
    jobject,
    jstring model_path,
    jint context_size,
    jint threads
) {
    static std::once_flag backend_once;
    std::call_once(backend_once, [] { llama_backend_init(); });

    const std::string path = jstring_to_string(env, model_path);
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    auto * engine = new Engine();
    engine->context_size = context_size;
    engine->model = llama_model_load_from_file(path.c_str(), model_params);
    if (engine->model == nullptr) {
        delete engine;
        throw_error(env, "Unable to load Hy-MT2 model. Confirm the Q4_K_M GGUF file is complete and compatible.");
        return 0;
    }

    engine->vocab = llama_model_get_vocab(engine->model);
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = context_size;
    context_params.n_batch = context_size;
    context_params.n_ubatch = context_size;
    context_params.n_threads = threads;
    context_params.n_threads_batch = threads;

    engine->context = llama_init_from_model(engine->model, context_params);
    if (engine->context == nullptr) {
        delete engine;
        throw_error(env, "Unable to create the Hy-MT2 context. Try a smaller context size or close other apps.");
        return 0;
    }

    auto sampler_params = llama_sampler_chain_default_params();
    engine->sampler = llama_sampler_chain_init(sampler_params);
    // Translation should be deterministic; sampling causes avoidable additions
    // such as invented dialogue after the requested source line.
    llama_sampler_chain_add(engine->sampler, llama_sampler_init_greedy());

    return reinterpret_cast<jlong>(engine);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_kanjilens_translate_HyMt2Engine_nativeTranslate(
    JNIEnv * env,
    jobject,
    jlong handle,
    jstring prompt,
    jint max_tokens,
    jobject listener
) {
    auto * engine = reinterpret_cast<Engine *>(handle);
    if (engine == nullptr) {
        throw_error(env, "Hy-MT2 engine is not loaded.");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);
    jclass listener_class = env->GetObjectClass(listener);
    jmethodID on_token = env->GetMethodID(listener_class, "onToken", "(Ljava/lang/String;)V");
    if (on_token == nullptr) {
        throw_error(env, "Hy-MT2 streaming callback is unavailable.");
        return nullptr;
    }

    const std::string prompt_text = format_hymt2_chat_prompt(jstring_to_string(env, prompt));
    auto tokens = tokenize(engine->vocab, prompt_text);
    if (tokens.empty() || static_cast<int>(tokens.size()) >= engine->context_size) {
        throw_error(env, "The translation request is too long for the local Hy-MT2 context.");
        return nullptr;
    }

    llama_memory_clear(llama_get_memory(engine->context), true);
    llama_sampler_reset(engine->sampler);

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(engine->context, batch) != 0) {
        throw_error(env, "Hy-MT2 failed while evaluating the translation prompt.");
        return nullptr;
    }

    std::string output;
    for (int generated = 0; generated < max_tokens; ++generated) {
        llama_token token = llama_sampler_sample(engine->sampler, engine->context, -1);
        if (llama_vocab_is_eog(engine->vocab, token)) break;

        char piece[512];
        // Do not render control tokens in translated text. The model may emit
        // a ChatML turn delimiter before llama.cpp classifies it as EOG.
        const int length = llama_token_to_piece(engine->vocab, token, piece, sizeof(piece), 0, false);
        if (length < 0) {
            throw_error(env, "Hy-MT2 returned an invalid output token.");
            return nullptr;
        }
        const size_t output_size_before_piece = output.size();
        output.append(piece, length);

        // Stop before the next chat turn or prompt delimiter. Hy-MT2 can echo
        // </source> after a valid translation if generation continues.
        size_t stop_at = output.size();
        for (const char * delimiter : {
                "<|im_end|>", "<|im_start|>", "<source>", "</source>",
                "<code>", "</code>", "<font", "</font>"
            }) {
            const size_t position = output.find(delimiter);
            if (position != std::string::npos) stop_at = std::min(stop_at, position);
        }
        if (stop_at != output.size()) {
            output.erase(stop_at);
            break;
        }

        if (output.size() > output_size_before_piece) {
            const std::string emitted = output.substr(output_size_before_piece);
            jstring emitted_java = env->NewStringUTF(emitted.c_str());
            env->CallVoidMethod(listener, on_token, emitted_java);
            env->DeleteLocalRef(emitted_java);
            if (env->ExceptionCheck()) return nullptr;
        }

        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(engine->context, batch) != 0) {
            throw_error(env, "Hy-MT2 failed while generating the translation.");
            return nullptr;
        }
    }

    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_kanjilens_translate_HyMt2Engine_nativeDestroy(JNIEnv *, jobject, jlong handle) {
    delete reinterpret_cast<Engine *>(handle);
}
