package tech.qdrant.glasses.search

interface SpeechRecognizer {
    fun startListening(onPartial: (String) -> Unit, onResult: (String) -> Unit, onError: (String) -> Unit)
    fun stopListening()
    fun destroy()
}
