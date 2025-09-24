package com.example.skills;

import com.example.skills.datatypes.AggregateScoreRequest;
import com.example.skills.datatypes.PromptContext;
import com.example.skills.datatypes.ContextPromptType;
import ingest.ChatPromptSystemContext;
import match.AggregateGroupScoreType;
import match.AggregateScoringAlgorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/resume-match")
public class ResumeAgentTuningController {
    Logger logger = LoggerFactory.getLogger(ResumeAgentTuningController.class);

    @Autowired
    @Qualifier("githubPromptSystemContext")
    private ChatPromptSystemContext githubPromptSystemContext;

    @Autowired
    @Qualifier("linkedInPromptSystemContext")
    private ChatPromptSystemContext linkedInPromptSystemContext;

    @Autowired
    @Qualifier("skillsQueryPrompt")
    private ChatPromptSystemContext skillsQueryPrompt;

    @Autowired
    private AggregateScoringAlgorithm aggregateScoringAlgorithm;

    @PostMapping("/context/set/{sessionId}")
    public ResponseEntity<Void> setPromptContext(@RequestBody PromptContext request,
                                                 @PathVariable String sessionId,
                                                 @RequestParam(required = true) String type) {
        ContextPromptType contextPromptType = ContextPromptType.fromString(type);
        if (contextPromptType == null) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
        }
        switch (contextPromptType) {
            case GITHUB -> githubPromptSystemContext.setSystemContext(request.context());
            case LINKEDIN -> linkedInPromptSystemContext.setSystemContext(request.context());
            case SKILLSQUERY -> skillsQueryPrompt.setSystemContext(request.context());
        }
        return ResponseEntity.ok().build();
    }

    @GetMapping("/context/get/{sessionId}")
    public ResponseEntity<String> getPromptContext(@PathVariable String sessionId,
                                                   @RequestParam(required = true) String type) {
        ContextPromptType contextPromptType = ContextPromptType.fromString(type);
        if (contextPromptType == null) {
            return ResponseEntity.badRequest().body("Unknown prompt context type: " + type);
        }
        String context = switch (contextPromptType) {
            case GITHUB -> githubPromptSystemContext.getSystemContext();
            case LINKEDIN -> linkedInPromptSystemContext.getSystemContext();
            case SKILLSQUERY -> skillsQueryPrompt.getSystemContext();
        };
        return ResponseEntity.ok(context);
    }

    @PutMapping("/aggregate-score/set/{sessionId}")
    public ResponseEntity<Void> setAggregateGroupScore(@RequestBody AggregateScoreRequest request,
                                                       @PathVariable String sessionId) {
        String scoreType = request.getScore();
        AggregateGroupScoreType aggregateGroupScoreType = AggregateGroupScoreType.fromString(scoreType);
        if (aggregateGroupScoreType == null) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
        }
        aggregateScoringAlgorithm.setScoringAlgorithm(scoreType);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/aggregate-score/get/{sessionId}")
    public ResponseEntity<String> getAggregateGroupScore(@PathVariable String sessionId) {
        return ResponseEntity.ok(aggregateScoringAlgorithm.getScoringAlgorithm());
    }
}
