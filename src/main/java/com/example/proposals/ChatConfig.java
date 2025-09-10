package com.example.proposals;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

//@Configuration
public class ChatConfig {
    @Bean
    public ChatClient.Builder chatClientBuilder(ChatModel chatModel) {
        ChatClient.Builder builder = ChatClient.builder(chatModel);
        return builder;
    }
}
