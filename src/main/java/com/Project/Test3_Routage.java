package com.Project;


import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test3_Routage {

    public static void main(String[] args) throws Exception {

        configureLogger();

        String llmKey = System.getenv("GEMINI_API_KEY");
        ChatLanguageModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // ============ PHASE 1 : INGESTION DE 2 DOCUMENTS ============
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Document 1 : RAG
        URL fileUrl1 = Test3_Routage.class.getResource("/rag.pdf");
        Path path1 = Paths.get(fileUrl1.toURI());
        EmbeddingStore<TextSegment> embeddingStore1 = ingestDocument(path1, embeddingModel);

        // Document 2 : Autre sujet (remplacez par votre fichier)
        URL fileUrl2 = Test3_Routage.class.getResource("/autre-document.pdf");
        Path path2 = Paths.get(fileUrl2.toURI());
        EmbeddingStore<TextSegment> embeddingStore2 = ingestDocument(path2, embeddingModel);

        System.out.println(" Phase d'ingestion des 2 documents terminée !");

        // ============ PHASE 2 : CRÉATION DES CONTENT RETRIEVERS ============
        ContentRetriever contentRetriever1 = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore1)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ContentRetriever contentRetriever2 = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore2)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // ============ PHASE 3 : ROUTAGE AVEC LE LM ============
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(contentRetriever1,
                "Support de cours sur le RAG (Retrieval Augmented Generation) " +
                        "et le fine-tuning en intelligence artificielle");
        descriptions.put(contentRetriever2,
                "\"Support de cours sur LangChain4j : présentation, modèles, \" +\n" +
                "\"AiServices, extraction de données, outils, modération et streaming\"");

        QueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // ============ PHASE 4 : CRÉATION DE L'ASSISTANT ============
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // ============ POSER DES QUESTIONS ============
        Scanner scanner = new Scanner(System.in);
        System.out.println("\n Assistant RAG avec Routage prêt !");
        System.out.println("   Le LM choisira automatiquement le bon document.\n");

        while (true) {
            System.out.print("Question : ");
            String question = scanner.nextLine();

            if ("quit".equalsIgnoreCase(question.trim())) {
                System.out.println("Au revoir !");
                break;
            }

            if (question.trim().isEmpty()) {
                continue;
            }

            String reponse = assistant.chat(question);
            System.out.println("\n Réponse : " + reponse + "\n");
        }

        scanner.close();
    }

    private static EmbeddingStore<TextSegment> ingestDocument(
            Path path,
            EmbeddingModel embeddingModel) throws Exception {

        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(600, 0);
        List<TextSegment> segments = splitter.split(document);

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        return embeddingStore;
    }

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);

        System.out.println(" Logging activé !\n");
    }
}
