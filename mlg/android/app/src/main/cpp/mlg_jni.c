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

/* ---- mind: the decisions, shared with the lab (engine/mlg_mind.c) ----------------- */
static char *jstr(JNIEnv *env, jstring s) {
    if (!s) return strdup("");
    const char *u = (*env)->GetStringUTFChars(env, s, NULL);
    char *out = strdup(u);
    (*env)->ReleaseStringUTFChars(env, s, u);
    return out;
}

JNIEXPORT jlong JNICALL FN(mindNew)(JNIEnv *env, jclass cls, jstring memory) {
    char *p = jstr(env, memory);
    mlg_mind *m = mlg_mind_new(p);
    free(p);
    return (jlong)(intptr_t)m;
}

JNIEXPORT void JNICALL FN(mindFree)(JNIEnv *env, jclass cls, jlong m) { mlg_mind_free((mlg_mind *)(intptr_t)m); }

JNIEXPORT void JNICALL FN(mindOrgans)(JNIEnv *env, jclass cls, jlong m, jlong router, jlong know, jlong core) {
    mlg_mind_organs((mlg_mind *)(intptr_t)m, (mlg_router *)(intptr_t)router, (mlg_know *)(intptr_t)know,
                    (mlg_core *)(intptr_t)core);
}

JNIEXPORT void JNICALL FN(mindBody)(JNIEnv *env, jclass cls, jlong m, jstring facts) {
    char *f = jstr(env, facts);
    mlg_mind_body((mlg_mind *)(intptr_t)m, f);
    free(f);
}

JNIEXPORT void JNICALL FN(mindLab)(JNIEnv *env, jclass cls, jlong m, jstring status) {
    char *f = jstr(env, status);
    mlg_mind_lab((mlg_mind *)(intptr_t)m, f);
    free(f);
}

JNIEXPORT void JNICALL FN(mindClock)(JNIEnv *env, jclass cls, jlong m, jlong unix_s, jint tz) {
    mlg_mind_clock((mlg_mind *)(intptr_t)m, unix_s, tz);
}

/* the thought lives on the native heap until thoughtFree */
JNIEXPORT jlong JNICALL FN(mindThink)(JNIEnv *env, jclass cls, jlong m, jbyteArray text) {
    char *t = cstr(env, text, NULL);
    mlg_thought *th = malloc(sizeof *th);
    mlg_mind_think((mlg_mind *)(intptr_t)m, t, th);
    free(t);
    return (jlong)(intptr_t)th;
}

JNIEXPORT void JNICALL FN(thoughtFree)(JNIEnv *env, jclass cls, jlong t) { free((void *)(intptr_t)t); }

JNIEXPORT jbyteArray JNICALL FN(thoughtJson)(JNIEnv *env, jclass cls, jlong m, jlong t) {
    char *buf = malloc(16384);
    const int n = mlg_mind_thought_json((mlg_mind *)(intptr_t)m, (mlg_thought *)(intptr_t)t, buf, 16384);
    jbyteArray a = bytes_out(env, buf, n < 16384 ? n : 16383);
    free(buf);
    return a;
}

JNIEXPORT jbyteArray JNICALL FN(thoughtPrompt)(JNIEnv *env, jclass cls, jlong t) {
    const mlg_thought *th = (mlg_thought *)(intptr_t)t;
    return bytes_out(env, th->prompt, (int)strlen(th->prompt));
}

JNIEXPORT jbyteArray JNICALL FN(mindSpeak)(JNIEnv *env, jclass cls, jlong m, jlong t, jlong seed, jobject sink) {
    char out[4096];
    sink_ctx s = {env, sink, NULL};
    if (sink) s.on_byte = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, sink), "onByte", "(I)Z");
    mlg_mind_speak((mlg_mind *)(intptr_t)m, (mlg_thought *)(intptr_t)t, out, sizeof out, to_java, &s, (uint64_t)seed);
    return bytes_out(env, out, (int)strlen(out));
}

JNIEXPORT jbyteArray JNICALL FN(mindFinish)(JNIEnv *env, jclass cls, jlong m, jlong t, jbyteArray raw) {
    char *r = cstr(env, raw, NULL), out[4096];
    mlg_mind_finish((mlg_mind *)(intptr_t)m, (mlg_thought *)(intptr_t)t, r, out, sizeof out);
    free(r);
    return bytes_out(env, out, (int)strlen(out));
}

JNIEXPORT jstring JNICALL FN(mindSpoke)(JNIEnv *env, jclass cls, jlong m) {
    return (*env)->NewStringUTF(env, mlg_mind_spoke((mlg_mind *)(intptr_t)m));
}

JNIEXPORT jobjectArray JNICALL FN(mindMemory)(JNIEnv *env, jclass cls, jlong m) {
    mlg_mind *mind = (mlg_mind *)(intptr_t)m;
    const int n = mlg_mind_memory_count(mind);
    jobjectArray a = (*env)->NewObjectArray(env, n, (*env)->FindClass(env, "java/lang/String"), NULL);
    for (int i = 0; i < n; i++)
        (*env)->SetObjectArrayElement(env, a, i, (*env)->NewStringUTF(env, mlg_mind_memory_text(mind, i)));
    return a;
}
