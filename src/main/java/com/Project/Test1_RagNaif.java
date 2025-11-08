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
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class Test1_RagNaif {

    public static void main(String[] args) throws Exception {

        // ============ Création du ChatModel ============
        String llmKey = System.getenv("GEMINI_API_KEY");
        ChatLanguageModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .build();

        // ============ PHASE 1 : INGESTION ============

        // 1. Récupérer le Path du fichier PDF
        URL fileUrl = Test1_RagNaif.class.getResource("/rag.pdf");
        Path path = Paths.get(fileUrl.toURI());

        // 2. Créer le parser PDF
        DocumentParser parser = new ApacheTikaDocumentParser();

        // 3. Charger le document
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);

        // 4. Découper le document en morceaux (chunks)
        DocumentSplitter splitter = DocumentSplitters.recursive(600, 0);
        List<TextSegment> segments = splitter.split(document);

        // 5. Créer le modèle d'embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 6. Créer les embeddings pour tous les segments
        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        // 7. Stocker dans un magasin d'embeddings
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        System.out.println(" Phase d'ingestion terminée !");
        System.out.println("   Nombre de segments créés : " + segments.size());

        // ============ PHASE 2 : RÉCUPÉRATION ET GÉNÉRATION ============

        // 8. Créer le ContentRetriever
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)              // 2 résultats les plus pertinents
                .minScore(0.5)              // Score minimum de 0.5
                .build();

        // 9. Créer une mémoire pour la conversation
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // 10. Créer l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        // 11. Poser des questions en boucle
        Scanner scanner = new Scanner(System.in);
        System.out.println("\n Assistant RAG prêt !");
        System.out.println("   Tapez 'quit' pour quitter.\n");

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
}
