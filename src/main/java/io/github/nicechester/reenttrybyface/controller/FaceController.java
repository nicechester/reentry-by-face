package io.github.nicechester.reenttrybyface.controller;

import io.github.nicechester.reenttrybyface.service.FaceRecognition;
import io.micrometer.common.util.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.UUID;

@Controller
public class FaceController {
    @Autowired
    private FaceRecognition faceService;

    @GetMapping("/register")
    public String home() {
        return "register";
    }

    @GetMapping("/")
    public String showRegisterPage() {
        return "register";
    }

    @PostMapping("/register")
    public String registerFace(@RequestParam("name") String name, @RequestParam("faceImage") MultipartFile file, Model model) {
        File tempFile = null;
        try {
            // Validate file
            if (file.isEmpty()) {
                model.addAttribute("message", "Please select an image");
                return "register";
            }

            // Convert MultipartFile to File
            tempFile = convertMultipartFileToFile(file);

            // Debug: Log file info
            System.out.println("Received file: " + file.getOriginalFilename());
            System.out.println("File size: " + file.getSize() + " bytes");
            System.out.println("Temp file path: " + tempFile.getAbsolutePath());

            // Register the face
            faceService.register(tempFile, name);
            model.addAttribute("message", "Face registered for today! Name: " + name);
        } catch (Exception e) {
            model.addAttribute("message", "Error registering face: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Clean up temp file
            if (tempFile != null && tempFile.exists()) {
                tempFile.delete();
            }
        }
        return "register";
    }

    @GetMapping("/reenter")
    public String showReenterPage() {
        return "reenter";
    }

    @PostMapping("/reenter")
    public String reenter(@RequestParam("faceImage") MultipartFile file, Model model) {
        File tempFile = null;
        try {
            // Convert MultipartFile to File
            tempFile = convertMultipartFileToFile(file);

            // Recognize the face
            String name = faceService.recognize(tempFile);
            model.addAttribute("result",
                    StringUtils.isNotEmpty(name) ? "Face recognized. Welcome back " + name : "Face not recognized for today.");
        } catch (Exception e) {
            model.addAttribute("result", "Error recognizing face: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Clean up temp file
            if (tempFile != null && tempFile.exists()) {
                tempFile.delete();
            }
        }
        return "reenter";
    }

    /**
     * Helper method to convert MultipartFile to temporary File
     */
    private File convertMultipartFileToFile(MultipartFile multipartFile) throws IOException {
        // Get original filename and extension
        String originalFilename = multipartFile.getOriginalFilename();
        String extension = "";
        if (originalFilename != null && originalFilename.contains(".")) {
            extension = originalFilename.substring(originalFilename.lastIndexOf("."));
        }

        // Create temp file
        Path tempFile = Files.createTempFile("face_", extension);

        // Copy MultipartFile content to temp file
        Files.copy(multipartFile.getInputStream(), tempFile, StandardCopyOption.REPLACE_EXISTING);

        return tempFile.toFile();
    }
}