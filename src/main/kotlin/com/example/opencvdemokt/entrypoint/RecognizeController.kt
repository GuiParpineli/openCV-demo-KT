package com.example.opencvdemokt.entrypoint

import com.example.opencvdemokt.service.MatchResponse
import com.example.opencvdemokt.service.RecognitionService
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.io.IOException

@RestController
@RequestMapping("/recognize")
class RecognizeController(private val service: RecognitionService) {

    @PostMapping(consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun compareImages(
        @RequestParam("targetImage") targetImage: MultipartFile,
        @RequestParam("queryImage") queryImage: MultipartFile
    ): ResponseEntity<MatchResponse> {
        try {
            val target = loadFromBytesInMemory(targetImage.bytes, targetImage.originalFilename)
            val query = loadFromBytesInMemory(queryImage.bytes, queryImage.originalFilename)
            val queryFeatures: Mat? = service.extractFeatures(query, query)
            val targetFeatures: Mat? = service.extractFeatures(target, target)
            return ResponseEntity.ok(service.matchFeatures(targetFeatures, queryFeatures))
        } catch (_: IOException) {
            return ResponseEntity.badRequest().body(MatchResponse(0.0, false))
        }
    }

    fun loadFromBytesInMemory(bytes: ByteArray, fileName: String?): Mat {
        require(bytes.isNotEmpty()) { "Arquivo vazio ou inválido: $fileName" }
        val mob = MatOfByte(*bytes)
        val image = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR)
        mob.release()
        require(!image.empty()) { "Não foi possível decodificar a imagem: $fileName" }
        return image
    }

}