package com.example.opencvdemokt.service

import org.bytedeco.javacpp.Loader
import org.bytedeco.opencv.opencv_java
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.objdetect.FaceDetectorYN
import org.opencv.objdetect.FaceRecognizerSF
import org.springframework.core.io.ClassPathResource
import org.springframework.core.io.Resource
import org.springframework.stereotype.Service
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

@Service
class RecognitionService {
    private val sFaceModel: String = extractModelToTempFile("models/face_recognition_sface_2021dec.onnx")!!
    private val yunetPath: String = extractModelToTempFile("models/face_detection_yunet_2023mar.onnx")!!

    companion object {
        private const val THRESHOLD_COSINE: Double = 0.36
        private const val DETECT_THRESHOLD: Float = 0.5f
        private const val NMS_THRESHOLD: Float = 0.3f
    }

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

    private val faceDetector: FaceDetectorYN = FaceDetectorYN.create(
        yunetPath,
        "",
        Size(320.0, 320.0),
        DETECT_THRESHOLD,
        NMS_THRESHOLD,
        1,
        Dnn.DNN_BACKEND_OPENCV,
        Dnn.DNN_TARGET_CPU
    )

    private fun infer(image: Mat): Mat {
        faceDetector.inputSize = image.size()
        val result = Mat()
        faceDetector.detect(image, result)
        return result
    }

    private fun extractFeatures(origImage: Mat?, faceImage: Mat?): Mat? {
        val targetAligned = Mat()
        recognizer.alignCrop(origImage, faceImage, targetAligned)
        val targetFeatures = Mat()
        recognizer.feature(targetAligned, targetFeatures)
        return targetFeatures.clone()
    }

    fun matchFeatures(target: Mat, query: Mat): MatchResponse {
        val queryFace = infer(query).row(0)
        val queryFeatures = this.extractFeatures(query, queryFace)
        val targetFace = infer(target).row(0)
        val targetFeatures = this.extractFeatures(target, targetFace)
        val distanceType = 0
        val score = recognizer.match(targetFeatures, queryFeatures, distanceType)
        val isMatch: Boolean = score >= THRESHOLD_COSINE
        return MatchResponse(score, isMatch)
    }

    private fun extractModelToTempFile(modelFilePath: String): String? {
        try {
            val resource: Resource = ClassPathResource(modelFilePath)
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