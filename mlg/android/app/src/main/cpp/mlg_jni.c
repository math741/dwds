/* JNI bridge: ai.mlg.zero.Native <-> the MLG engine (mlg/engine). */
#include <jni.h>
#include <stdlib.h>
#include <string.h>

#include "mlg.h"

static char last_error[256];

#define FN(name) Java_ai_mlg_zero_Native_##name

static jbyteArray bytes_out(JNIEnv *env, const void *p, int n) {
    jbyteArray a = (*env)->NewByteArray(env, n);
    if (a && n) (*env)->SetByteArrayRegion(env, a, 0, n, (const jbyte *)p);
    return a;
}

/* a NUL-terminated copy of a Java byte[] (UTF-8 text) */
static char *cstr(JNIEnv *env, jbyteArray a, int *len) {
    const int n = a ? (*env)->GetArrayLength(env, a) : 0;
    char *s = malloc(n + 1);
    if (n) (*env)->GetByteArrayRegion(env, a, 0, n, (jbyte *)s);
    s[n] = 0;
    if (len) *len = n;
    return s;
}

static char *path_of(JNIEnv *env, jstring p) {
    const char *u = (*env)->GetStringUTFChars(env, p, NULL);
    char *s = strdup(u);
    (*env)->ReleaseStringUTFChars(env, p, u);
    return s;
}

JNIEXPORT jstring JNICALL FN(lastError)(JNIEnv *env, jclass cls) { return (*env)->NewStringUTF(env, last_error); }
JNIEXPORT void JNICALL FN(setThreads)(JNIEnv *env, jclass cls, jint n) { mlg_set_threads(n); }
JNIEXPORT jint JNICALL FN(getThreads)(JNIEnv *env, jclass cls) { return mlg_get_threads(); }

/* ---- core ----------------------------------------------------------------- */
JNIEXPORT jlong JNICALL FN(coreOpen)(JNIEnv *env, jclass cls, jstring path) {
    char *p = path_of(env, path);
    mlg_core *c = mlg_core_open(p, last_error, sizeof last_error);
    free(p);
    return (jlong)(intptr_t)c;
}

JNIEXPORT void JNICALL FN(coreClose)(JNIEnv *env, jclass cls, jlong h) { mlg_core_close((mlg_core *)(intptr_t)h); }

JNIEXPORT jlongArray JNICALL FN(coreInfo)(JNIEnv *env, jclass cls, jlong h) {
    const mlg_core_info i = mlg_core_get_info((mlg_core *)(intptr_t)h);
    const jlong v[9] = {i.d_model, i.n_layers, i.n_heads, i.n_experts, i.top_k, i.ffn_mult, i.ctx, i.wtype, i.params};
    jlongArray a = (*env)->NewLongArray(env, 9);
    (*env)->SetLongArrayRegion(env, a, 0, 9, v);
    return a;
}

JNIEXPORT jdoubleArray JNICALL FN(coreBench)(JNIEnv *env, jclass cls, jlong h) {
    double v[2];
    mlg_core_bench((mlg_core *)(intptr_t)h, &v[0], &v[1]);
    jdoubleArray a = (*env)->NewDoubleArray(env, 2);
    (*env)->SetDoubleArrayRegion(env, a, 0, 2, v);
    return a;
}

JNIEXPORT jdouble JNICALL FN(coreBitsPerByte)(JNIEnv *env, jclass cls, jlong h, jbyteArray text) {
    int n;
    char *t = cstr(env, text, &n);
    const double bpb = mlg_core_bits_per_byte((mlg_core *)(intptr_t)h, (const uint8_t *)t, n);
    free(t);
    return bpb;
}

typedef struct {
    JNIEnv *env;
    jobject sink;
    jmethodID on_byte;
} sink_ctx;

static int to_java(uint8_t b, void *user) {
    sink_ctx *s = user;
    if (!s->sink) return 0;
    return (*s->env)->CallBooleanMethod(s->env, s->sink, s->on_byte, (jint)b) ? 1 : 0;
}

/* sink: an object with `boolean onByte(int b)`; returning true stops generation */
JNIEXPORT jbyteArray JNICALL FN(coreGenerate)(JNIEnv *env, jclass cls, jlong h, jbyteArray prompt, jint max_out,
                                              jfloat temperature, jfloat top_p, jlong seed, jobject sink) {
    int n;
    char *p = cstr(env, prompt, &n);
    uint8_t *out = malloc(max_out > 0 ? max_out : 1);
    sink_ctx s = {env, sink, NULL};
    if (sink) s.on_byte = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, sink), "onByte", "(I)Z");
    const mlg_sample cfg = {temperature, top_p, (uint64_t)seed, 1, 4, 1};
    const int got = mlg_core_generate((mlg_core *)(intptr_t)h, (const uint8_t *)p, n, out, max_out, &cfg,
                                      to_java, &s);
    jbyteArray a = bytes_out(env, out, got);
    free(p), free(out);
    return a;
}

/* ---- retrieval ------------------------------------------------------------ */
JNIEXPORT jlong JNICALL FN(knowOpen)(JNIEnv *env, jclass cls, jstring path) {
    char *p = path_of(env, path);
    mlg_know *k = mlg_know_open(p, last_error, sizeof last_error);
    free(p);
    return (jlong)(intptr_t)k;
}

JNIEXPORT void JNICALL FN(knowClose)(JNIEnv *env, jclass cls, jlong h) { mlg_know_close((mlg_know *)(intptr_t)h); }
JNIEXPORT jint JNICALL FN(knowCount)(JNIEnv *env, jclass cls, jlong h) { return mlg_know_count((mlg_know *)(intptr_t)h); }

/* ids and scores interleaved as floats: [id0, score0, id1, score1, ...] */
JNIEXPORT jfloatArray JNICALL FN(knowSearch)(JNIEnv *env, jclass cls, jlong h, jbyteArray query, jint top) {
    char *q = cstr(env, query, NULL);
    int ids[64];
    float sc[64], out[128];
    if (top > 64) top = 64;
    const int n = mlg_know_search((mlg_know *)(intptr_t)h, q, top, ids, sc);
    for (int i = 0; i < n; i++) out[2 * i] = (float)ids[i], out[2 * i + 1] = sc[i];
    free(q);
    jfloatArray a = (*env)->NewFloatArray(env, 2 * n);
    (*env)->SetFloatArrayRegion(env, a, 0, 2 * n, out);
    return a;
}

JNIEXPORT jbyteArray JNICALL FN(knowText)(JNIEnv *env, jclass cls, jlong h, jint id) {
    int n;
    const char *t = mlg_know_text((mlg_know *)(intptr_t)h, id, &n);
    return bytes_out(env, t, n);
}

JNIEXPORT jbyteArray JNICALL FN(knowTitle)(JNIEnv *env, jclass cls, jlong h, jint id) {
    int n;
    const char *t = mlg_know_title((mlg_know *)(intptr_t)h, id, &n);
    return bytes_out(env, t, n);
}

/* ---- router --------------------------------------------------------------- */
JNIEXPORT jlong JNICALL FN(routerOpen)(JNIEnv *env, jclass cls, jstring path) {
    char *p = path_of(env, path);
    mlg_router *r = mlg_router_open(p, last_error, sizeof last_error);
    free(p);
    return (jlong)(intptr_t)r;
}

JNIEXPORT void JNICALL FN(routerClose)(JNIEnv *env, jclass cls, jlong h) { mlg_router_close((mlg_router *)(intptr_t)h); }

JNIEXPORT jobjectArray JNICALL FN(routerNames)(JNIEnv *env, jclass cls, jlong h) {
    mlg_router *r = (mlg_router *)(intptr_t)h;
    const int n = mlg_router_count(r);
    jobjectArray a = (*env)->NewObjectArray(env, n, (*env)->FindClass(env, "java/lang/String"), NULL);
    for (int i = 0; i < n; i++) (*env)->SetObjectArrayElement(env, a, i, (*env)->NewStringUTF(env, mlg_router_name(r, i)));
    return a;
}

JNIEXPORT jfloatArray JNICALL FN(routerRoute)(JNIEnv *env, jclass cls, jlong h, jbyteArray text) {
    mlg_router *r = (mlg_router *)(intptr_t)h;
    char *t = cstr(env, text, NULL);
    float p[256];
    mlg_router_route(r, t, p);
    free(t);
    const int n = mlg_router_count(r);
    jfloatArray a = (*env)->NewFloatArray(env, n);
    (*env)->SetFloatArrayRegion(env, a, 0, n, p);
    return a;
}
