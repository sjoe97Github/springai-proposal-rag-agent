package com.example.proposals;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/proposal")
public class ProposalController {

    @Value("classpath:/proposals-prompt-template.txt")
    private Resource ragPromptTemplate;

    @Value("${spring.ai.vectorstore.topk:2}")
    private int topK;

    private final ChatClient aiClient;
    private final VectorStore vectorStore;

    public ProposalController(ChatClient aiClient, VectorStore vectorStore) {
        this.aiClient = aiClient;
        this.vectorStore = vectorStore;
    }
//
//    @PostMapping
//    public Answer ask(@RequestBody Question question) {
//        String answer = aiClient.prompt()
//            .user(question.question())
//            .advisors(new QuestionAnswerAdvisor(vectorStore))
//            .call()
//            .content();
//
//        return new Answer(answer);
//    }

    @PostMapping("/generate")
    public String generateProposal(@RequestBody Question question) {
        SearchRequest searchRequest = SearchRequest.builder()
                .topK(topK)
                .query(question.question())
                .build();

        // TODO - It is theoretically possible for similarity search to return null or empty results, depending on vector store implementation.
        //        Check for null or empty results and handle accordingly.
        List<Document> similarDocuments = vectorStore.similaritySearch(searchRequest);
        List<String> contentList = similarDocuments.stream().map(Document::getText).toList();

        PromptTemplate promptTemplate = new PromptTemplate(ragPromptTemplate);
        Map<String, Object> promptParameters = new HashMap<>();
        promptParameters.put("input", question.question());
        promptParameters.put("sample_proposals", String.join("\n", contentList));
        Prompt prompt = promptTemplate.create(promptParameters);

        // TODO - add logging
        System.out.println(prompt);

        ChatResponse response = aiClient.prompt(prompt).call().chatResponse();

        // TODO - response.getResult() may be null, check for null and handle accordingly.
        return new Answer(response.getResult().getOutput().getText()).answer();
    }

}
