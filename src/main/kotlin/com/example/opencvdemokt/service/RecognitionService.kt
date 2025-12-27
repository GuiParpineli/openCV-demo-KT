package com.example.opencvdemokt.service

import org.bytedeco.javacpp.Loader
import org.bytedeco.opencv.opencv_java
import org.opencv.core.Mat
import org.opencv.dnn.Dnn
import org.opencv.objdetect.FaceRecognizerSF
import org.springframework.core.io.ClassPathResource
import org.springframework.core.io.Resource
import org.springframework.stereotype.Service
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

@Service
class RecognitionService {
    var sFaceModel: String? = extractModelToTempFile()

    init {
        Loader.load(opencv_java::class.java)
    }

    private val recognizer: FaceRecognizerSF = FaceRecognizerSF
        .create(
            sFaceModel,
            "",
            Dnn.DNN_BACKEND_OPENCV,
            Dnn.DNN_TARGET_CPU
        )

    private fun extractFeatures(origImage: Mat?, faceImage: Mat?): Mat? {
        val targetAligned = Mat()
        recognizer.alignCrop(origImage, faceImage, targetAligned)
        val targetFeatures = Mat()
        recognizer.feature(targetAligned, targetFeatures)
        return targetFeatures.clone()
    }

    fun matchFeatures(target: Mat?, query: Mat?): MatchResponse {
        val queryFeatures: Mat? = this.extractFeatures(query, query)
        val targetFeatures: Mat? = this.extractFeatures(target, target)
        val distanceType = 0
        val matchThreshold = 0.36
        val score = recognizer.match(targetFeatures, queryFeatures, distanceType)
        val isMatch: Boolean = score >= matchThreshold
        return MatchResponse(score, isMatch)
    }

    private fun extractModelToTempFile(): String? {
        try {
            val resource: Resource = ClassPathResource("models/face_recognition_sface_2021dec.onnx")
            try {
                return resource.file.absolutePath
            } catch (e: IOException) {
                val tempFile = Files.createTempFile("cvcore_", "_face_recognition.onnx")
                tempFile.toFile().deleteOnExit()
                resource.inputStream.use { `in` ->
                    Files.copy(`in`, tempFile, StandardCopyOption.REPLACE_EXISTING)
                }
                return tempFile.toAbsolutePath().toString()
            }
        } catch (e: IOException) {
            throw RuntimeException("Falha ao carregar modelo SFace", e)
        }
    }
}