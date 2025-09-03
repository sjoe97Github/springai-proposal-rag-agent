package com.example.proposals;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

@Service
public class HypotheticalSearchStrategy {

    @Value("classpath:/hypothetical-prompt-template.txt")
    private Resource hypotheticalPromptTemplate;

    private final ChatClient chatClient;

    public HypotheticalSearchStrategy(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    public String generatePrompt(String query) {
        String hypotheticalPrompt = String.format(getHypotheticalPromptTemplate(), query);
        Prompt prompt = new Prompt(new UserMessage(hypotheticalPrompt));
        ChatResponse response = chatClient.prompt(prompt).call().chatResponse();
        return response == null ? hypotheticalPrompt : response.getResult().getOutput().getText();
    }

    private String getHypotheticalPromptTemplate() {
        try {
            return new String(hypotheticalPromptTemplate.getInputStream().readAllBytes());
        } catch (Exception e) {
            // TODO - Use a logging framework
            System.out.println("Failed to read hypothetical prompt template"+ e);

            return """
                    You are an expert in technical recruitment. Based on the following user query, write a detailed,
                    hypothetical resume summary that would be a perfect match. 
                    Focus on including technical skills and experience.
                    User Query: %s
                """;
        }
    }
}