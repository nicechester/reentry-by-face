package io.github.nicechester.reenttrybyface.service;

import io.micrometer.common.util.StringUtils;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.*;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

@Slf4j
@Service
public class FaceRecognition {

    private CascadeClassifier faceDetector;
    private Net faceNetModel;
    private static final double SIMILARITY_THRESHOLD = 0.9;
    private static final String DATA_FILE = "face_database.dat";

    // In-memory database of registered face embeddings
    private Map<String, float[]> registeredFaceEmbeddings;

    @PostConstruct
    public void init() {
        try {
            log.info("Initializing FaceNet recognition system...");

            // Load Haar Cascade for face detection
            InputStream cascadeStream = getClass().getClassLoader()
                    .getResourceAsStream("haarcascade_frontalface_default.xml");

            if (cascadeStream == null) {
                throw new RuntimeException("Cannot find haarcascade_frontalface_default.xml in classpath");
            }

            File cascadeTempFile = File.createTempFile("haarcascade", ".xml");
            cascadeTempFile.deleteOnExit();

            try (FileOutputStream out = new FileOutputStream(cascadeTempFile)) {
                byte[] buffer = new byte[1024];
                int bytesRead;
                while ((bytesRead = cascadeStream.read(buffer)) != -1) {
                    out.write(buffer, 0, bytesRead);
                }
            }
            cascadeStream.close();

            faceDetector = new CascadeClassifier(cascadeTempFile.getAbsolutePath());

            if (faceDetector.isNull() || faceDetector.empty()) {
                throw new RuntimeException("Failed to load cascade classifier");
            }

            // Load FaceNet model
            loadFaceNetModel();

            registeredFaceEmbeddings = new HashMap<>();
            loadDatabase();

            log.info("FaceNet recognition system initialized successfully");

        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize FaceRecognition", e);
        }
    }

    private void loadFaceNetModel() {
        try {
            log.info("Loading FaceNet model...");

            InputStream modelStream = getClass().getClassLoader()
                    .getResourceAsStream("nn4.small2.v1.t7");

            if (modelStream == null) {
                throw new RuntimeException("Cannot find nn4.small2.v1.t7 in classpath. " +
                        "Please download from: https://github.com/pyannote/pyannote-data/raw/master/openface.nn4.small2.v1.t7");
            }

            // Save to temp file
            File modelTempFile = File.createTempFile("facenet", ".t7");
            modelTempFile.deleteOnExit();

            try (FileOutputStream out = new FileOutputStream(modelTempFile)) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = modelStream.read(buffer)) != -1) {
                    out.write(buffer, 0, bytesRead);
                }
            }
            modelStream.close();

            // Load the model using Torch backend
            faceNetModel = readNetFromTorch(modelTempFile.getAbsolutePath());

            if (faceNetModel.empty()) {
                throw new RuntimeException("Failed to load FaceNet model");
            }

            log.info("FaceNet model loaded successfully");

        } catch (Exception e) {
            throw new RuntimeException("Failed to load FaceNet model: " + e.getMessage(), e);
        }
    }

    /**
     * Registers a face with an ID
     */
    public void register(File facePic, String name) {
        try {
            log.info("Registering face with ID: {}", name);

            Mat image = imread(facePic.getAbsolutePath());
            if (image.empty()) {
                throw new RuntimeException("Failed to load image: " + facePic.getAbsolutePath());
            }

            Mat face = detectFace(image);
            float[] embedding = getFaceEmbedding(face);

            registeredFaceEmbeddings.put(name, embedding);
            saveDatabase();

            log.info("Successfully registered face with ID: {}", name);
            log.info("Total registered faces: {}", registeredFaceEmbeddings.size());

        } catch (Exception e) {
            log.error("Error registering face: " + e.getMessage(), e);
            throw new RuntimeException("Error registering face: " + e.getMessage());
        }
    }

    /**
     * Recognizes a face and returns the matching ID
     */
    public String recognize(File facePic) {
        try {
            log.info("Attempting to recognize face...");

            Mat image = imread(facePic.getAbsolutePath());
            if (image.empty()) {
                throw new RuntimeException("Failed to load image: " + facePic.getAbsolutePath());
            }

            Mat face = detectFace(image);
            float[] targetEmbedding = getFaceEmbedding(face);

            String bestMatchId = "";
            double bestSimilarity = Double.MAX_VALUE;

            log.info("Comparing against {} registered faces (threshold: {})",
                    registeredFaceEmbeddings.size(), String.format("%.4f", SIMILARITY_THRESHOLD));

            for (Map.Entry<String, float[]> entry : registeredFaceEmbeddings.entrySet()) {
                String id = entry.getKey();
                float[] registeredEmbedding = entry.getValue();

                double distance = euclideanDistance(targetEmbedding, registeredEmbedding);

                log.info("  ID {}: distance={}", id, String.format("%.4f", distance));

                if (distance < SIMILARITY_THRESHOLD && distance < bestSimilarity) {
                    bestSimilarity = distance;
                    bestMatchId = id;
                }
            }

            if (StringUtils.isNotEmpty(bestMatchId)) {
                log.info("✓ MATCH FOUND! ID: {} (distance: {})", bestMatchId, String.format("%.4f", bestSimilarity));
            } else {
                log.info("✗ NO MATCH FOUND (best distance: {})",
                        registeredFaceEmbeddings.isEmpty() ? "N/A" : String.format("%.4f", bestSimilarity));
            }

            return bestMatchId;

        } catch (Exception e) {
            log.error("Error recognizing face: " + e.getMessage(), e);
            return "";
        }
    }

    /**
     * Detects face in an image
     */
    private Mat detectFace(Mat image) {
        Mat grayImage = new Mat();
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
        equalizeHist(grayImage, grayImage);

        RectVector faceDetections = new RectVector();

        faceDetector.detectMultiScale(
                grayImage,
                faceDetections,
                1.1,
                3,
                0,
                new Size(30, 30),
                new Size()
        );

        log.info("Face detection found {} faces", faceDetections.size());

        if (faceDetections.size() == 0) {
            // Try with more lenient parameters
            log.info("Trying with more lenient parameters...");
            faceDetector.detectMultiScale(
                    grayImage,
                    faceDetections,
                    1.05,
                    2,
                    0,
                    new Size(20, 20),
                    new Size()
            );
            log.info("Second attempt found {} faces", faceDetections.size());
        }

        if (faceDetections.size() == 0) {
            throw new RuntimeException("No face detected. Please ensure your face is clearly visible and well-lit.");
        }

        Rect faceRect = faceDetections.get(0);
        log.info("Detected face at: x={}, y={}, width={}, height={}",
                faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height());

        Mat face = new Mat(image, faceRect);

        // Resize to FaceNet input size
        Mat resizedFace = new Mat();
        Size size = new Size(96, 96);
        resize(face, resizedFace, size);

        return resizedFace;
    }

    /**
     * Gets face embedding using FaceNet
     */
    private float[] getFaceEmbedding(Mat face) {
        // Prepare input blob for FaceNet
        Mat blob = blobFromImage(face, 1.0 / 255, new Size(96, 96),
                new Scalar(0.0, 0.0, 0.0, 0.0), true, false, CV_32F);

        faceNetModel.setInput(blob);
        Mat output = faceNetModel.forward();

        // Convert to float array (128-dimensional embedding)
        float[] embedding = new float[(int) output.total()];
        FloatPointer fp = new FloatPointer(output.data());
        fp.get(embedding);

        // Normalize the embedding
        embedding = normalizeEmbedding(embedding);

        log.debug("Generated {}-dimensional face embedding", embedding.length);

        return embedding;
    }

    /**
     * Normalizes an embedding vector (L2 normalization)
     */
    private float[] normalizeEmbedding(float[] embedding) {
        double sum = 0;
        for (float v : embedding) {
            sum += v * v;
        }
        double norm = Math.sqrt(sum);

        if (norm > 0) {
            for (int i = 0; i < embedding.length; i++) {
                embedding[i] /= norm;
            }
        }

        return embedding;
    }

    /**
     * Calculates Euclidean distance between two embeddings
     */
    private double euclideanDistance(float[] embedding1, float[] embedding2) {
        if (embedding1.length != embedding2.length) {
            throw new IllegalArgumentException("Embeddings must have same length");
        }

        double sum = 0;
        for (int i = 0; i < embedding1.length; i++) {
            double diff = embedding1[i] - embedding2[i];
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }

    /**
     * Saves the face database to disk
     */
    private void saveDatabase() {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(DATA_FILE))) {

            oos.writeObject(registeredFaceEmbeddings);
            log.info("Database saved successfully");

        } catch (IOException e) {
            log.error("Error saving database: " + e.getMessage());
        }
    }

    /**
     * Loads the face database from disk
     */
    @SuppressWarnings("unchecked")
    private void loadDatabase() {
        File dbFile = new File(DATA_FILE);
        if (!dbFile.exists()) {
            log.info("No existing database found. Starting fresh.");
            return;
        }

        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(DATA_FILE))) {

            registeredFaceEmbeddings = (Map<String, float[]>) ois.readObject();
            log.info("Database loaded: {} faces registered", registeredFaceEmbeddings.size());

        } catch (IOException | ClassNotFoundException e) {
            log.error("Error loading database: " + e.getMessage());
            registeredFaceEmbeddings = new HashMap<>();
        }
    }

    public int getRegisteredCount() {
        return registeredFaceEmbeddings.size();
    }

    public void clearDatabase() {
        registeredFaceEmbeddings.clear();
        saveDatabase();
        log.info("Database cleared");
    }
}