# openCV-demo-KT

A demonstration of face recognition using OpenCV with Kotlin and Spring Boot.

## Overview

This project provides a simple REST API to compare two images and determine if they contain the same face using the SFace (Face Recognition) model.

## Tech Stack

- **Language:** Kotlin 2.3.0
- **Framework:** Spring Boot 4.0.1
- **OpenCV Wrapper:** [Bytedeco OpenCV](https://github.com/bytedeco/javacpp-presets/tree/master/opencv) 4.11.0-1.5.12
- **Model:** SFace (included in `src/main/resources/models`)
- **API Documentation:** SpringDoc OpenAPI (Swagger UI)

## Requirements

- **JDK:** 25
- **Package Manager:** Gradle (wrapper included)

## Getting Started


### Run the application
```bash
./gradlew bootRun
```
The application will start by default on port 8080.

### Swagger UI
Once running, you can access the API documentation and test the endpoints at:
[http://localhost:8080/swagger-ui.html](http://localhost:8080/swagger-ui.html)

## API Endpoints

### Recognize Faces
Compares a target image against a query image.

- **URL:** `/recognize`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
    - `targetImage`: The reference image (MultipartFile)
    - `queryImage`: The image to be checked (MultipartFile)

**Example Response:**
```json
{
  "score": 0.45,
  "isMatch": true
}
```

## Scripts

- `./gradlew bootRun`: Runs the Spring Boot application.
- `./gradlew build`: Builds the project and generates the JAR.
- `./gradlew test`: Runs unit tests.
- `./gradlew clean`: Cleans the build directory.

## Project Structure

```text
src/
├── main/
│   ├── kotlin/com/example/opencvdemokt/
│   │   ├── entrypoint/         # REST Controllers
│   │   ├── service/            # Business Logic & OpenCV Integration
│   │   └── OpenCvDemoKtApplication.kt # Main Entry Point
│   └── resources/
│       ├── models/             # SFace ONNX model
│       └── application.yaml    # Application configuration
└── test/                       # Unit and Integration tests
```
