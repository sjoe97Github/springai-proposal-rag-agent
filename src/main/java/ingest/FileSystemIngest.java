package ingest;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class FileSystemIngest implements IngestResources {

    private Resource documentResource;
    private boolean recursive;
    private String includeExts;

    public FileSystemIngest(Resource documentResource, boolean recursive, String includeExtensions) {
        this.documentResource = documentResource;
        this.recursive = recursive;
        this.includeExts = includeExtensions;
    }

    public List<Resource> getResources() throws IOException {
        return resolveFileResources(documentResource, recursive, parseExtensions(includeExts));
    }

    public List<Resource> resolveFileResources(Resource root, boolean recursive, Set<String> includeExts) throws IOException {
        // Prefer filesystem paths for directory scanning
        if (root.isFile() && root.getFile().isDirectory()) {
            Path dir = root.getFile().toPath();
            Stream<Path> walker = recursive ? Files.walk(dir) : Files.list(dir);
            try (walker) {
                return walker
                        .filter(Files::isRegularFile)
                        .filter(p -> includeExts.isEmpty() || includeExts.contains(getExtLower(p)))
                        .sorted()
                        .map(FileSystemResource::new)
                        .collect(Collectors.toList());
            }
        }

        // Single file (or a non-filesystem resource). Just return it as-is.
        return List.of(root);
    }

    private Set<String> parseExtensions(String extensions) {
        if (extensions == null || extensions.isBlank()) return Collections.emptySet();
        return Arrays.stream(extensions.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .map(s -> s.startsWith(".") ? s.substring(1) : s)
                .map(String::toLowerCase)
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }

    private String getExtLower(Path p) {
        String name = p.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot >= 0 ? name.substring(dot + 1).toLowerCase() : "";
    }

}
