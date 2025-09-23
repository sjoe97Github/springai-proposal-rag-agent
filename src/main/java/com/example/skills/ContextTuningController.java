package com.example.skills;

import com.example.skills.datatypes.PromptContext;
import com.example.skills.datatypes.PromptType;
import ingest.ChatPromptSystemContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/resume-match/context")
public class ContextTuningController {
    Logger logger = LoggerFactory.getLogger(ContextTuningController.class);

    @Autowired
    @Qualifier("githubPromptSystemContext")
    private ChatPromptSystemContext githubPromptSystemContext;

    @Autowired
    @Qualifier("linkedInPromptSystemContext")
    private ChatPromptSystemContext linkedInPromptSystemContext;

    @Autowired
    @Qualifier("skillsQueryPrompt")
    private ChatPromptSystemContext skillsQueryPrompt;

    @PostMapping("/set/{sessionId}")
    public ResponseEntity<Void> setPromptContext(@RequestBody PromptContext request,
                                                 @PathVariable String sessionId,
                                                 @RequestParam(required = true) String type) {
        PromptType promptType = PromptType.fromString(type);
        if (promptType == null) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
        }
        switch (promptType) {
            case GITHUB -> githubPromptSystemContext.setSystemContext(request.context());
            case LINKEDIN -> linkedInPromptSystemContext.setSystemContext(request.context());
            case SKILLSQUERY -> skillsQueryPrompt.setSystemContext(request.context());
        }
        return ResponseEntity.ok().build();
    }

    @GetMapping("/get/{sessionId}")
    public ResponseEntity<String> getPromptContext(@PathVariable String sessionId,
                                                   @RequestParam(required = true) String type) {
        PromptType promptType = PromptType.fromString(type);
        if (promptType == null) {
            return ResponseEntity.badRequest().body("Unknown prompt context type: " + type);
        }
        String context = switch (promptType) {
            case GITHUB -> githubPromptSystemContext.getSystemContext();
            case LINKEDIN -> linkedInPromptSystemContext.getSystemContext();
            case SKILLSQUERY -> skillsQueryPrompt.getSystemContext();
        };
        return ResponseEntity.ok(context);
    }
}
