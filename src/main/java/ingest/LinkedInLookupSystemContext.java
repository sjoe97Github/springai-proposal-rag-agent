package ingest;

public class LinkedInLookupSystemContext implements ChatPromptSystemContext {
    @Override
    public String getSystemContext() {
        return systemContext;
    }

    @Override
    public void setSystemContext(String systemContext) {
        this.systemContext = systemContext;
    }

    // Initial default system context
    // TODO - Future, lookup default from database or config file
    private String systemContext = """
                You are a LinkedIn talent recruiter assistant.

                You may call the "scan_profile" tool to capture LinkedIn user profile details.

                Use tool results to answer clearly and concisely.
                The tool returns a list of repositories in a JSON format similar to this example:
                [
                    profile-details: {
                        content: null
                    }
                ]
            """;
}
