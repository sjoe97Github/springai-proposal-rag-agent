package ingest;

import org.springframework.core.io.Resource;

import java.io.IOException;
import java.util.List;

public interface IngestResources {
    List<Resource> getResources() throws IOException;
}
