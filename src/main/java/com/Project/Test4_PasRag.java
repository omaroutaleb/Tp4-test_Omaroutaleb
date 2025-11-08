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
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test4_PasRag {

    public static void main(String[] args) throws Exception {

        configureLogger();

        String llmKey = System.getenv("GEMINI_API_KEY");
        ChatLanguageModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // ============ PHASE 1 : INGESTION ============
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        URL fileUrl = Test4_PasRag.class.getResource("/rag.pdf");
        Path path = Paths.get(fileUrl.toURI());

        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(600, 0);
        List<TextSegment> segments = splitter.split(document);

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        System.out.println(" Phase d'ingestion terminée !");

        // ============ PHASE 2 : QUERY ROUTER PERSONNALISÉ ============

        // Classe interne pour éviter le RAG si la question n'est pas sur l'IA
        class QueryRouterPourEviterRag implements QueryRouter {
            @Override
            public List<ContentRetriever> route(Query query) {
                // Créer un template de prompt
                PromptTemplate template = PromptTemplate.from(
                        "Est-ce que la requête '{{requete}}' porte sur l'IA " +
                                "(Intelligence Artificielle, RAG, Fine-tuning, embeddings, LLM) ? " +
                                "Réponds seulement par 'oui', 'non', ou 'peut-être'."
                );

                Prompt prompt = template.apply(Map.of("requete", query.text()));
                String reponse = model.generate(prompt.text());

                System.out.println("🔍 Décision du routeur : " + reponse);

                if (reponse.toLowerCase().contains("non")) {
                    // Pas de RAG
                    return Collections.emptyList();
                } else {
                    // Utiliser le RAG
                    return Collections.singletonList(contentRetriever);
                }
            }
        }

        QueryRouter queryRouter = new QueryRouterPourEviterRag();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // ============ PHASE 3 : CRÉATION DE L'ASSISTANT ============
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // ============ TESTS ============
        System.out.println("\n Test 1 : Question générale (pas de RAG attendu)");
        String reponse1 = assistant.chat("Bonjour");
        System.out.println(" Réponse : " + reponse1 + "\n");

        System.out.println(" Test 2 : Question sur le RAG (RAG attendu)");
        String reponse2 = assistant.chat("Qu'est-ce que le RAG ?");
        System.out.println(" Réponse : " + reponse2 + "\n");

        System.out.println(" Test 3 : Question hors contexte (pas de RAG attendu)");
        String reponse3 = assistant.chat("Quelle est la capitale de la France ?");
        System.out.println(" Réponse : " + reponse3);
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
