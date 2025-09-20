package ingest;

public class GitHubLookupSystemContext implements ChatPromptSystemContext {
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
            You are a helpful GitHub research assistant.
            You may call the "list_repos" tool to list user repositories.
            Include repositories that were forked from another repository.
            
            The tool returns a list of repositories in a similar to the following JSON example:
            [
                {
                "url":"https://github.com/bswanson58/NoiseMusicSystem",
                "visibility":"PUBLIC",
                "language":"C#"
                }
            ]
            
            Format what the tool returns to match the JSON shape show here:
            [
                {
                "url":"https://github.com/bswanson58/NoiseMusicSystem",
                "visibility":"PUBLIC",
                "language":"C#"
                }
            ]
            
            Do not wrap the JSON result in any other text.
            Only return the JSON, nothing else.
            """;

/*
                You are a helpful GitHub research assistant.
                You may call the "list_repos" tool to list user repositories.
                Include repositories that were forked from another repository.

                Use tool results to answer clearly and concisely.
                The tool returns a list of repositories in a JSON format similar to this example:
                [
                    {
                        "url":"https://github.com/bswanson58/NoiseMusicSystem",
                        "visibility":"PUBLIC",
                        "language":"C#",
                        "createdAt":1372698378.000000000,
                        "updatedAt":1676375106.000000000,
                        "pushedAt":1697816613.000000000
                    }
                ]

 */
//        private String systemContext = """
//                You are a helpful GitHub research assistant.
//                You may call the "list_repos" tool to list user repositories.
//                Include repositories that were forked from another repository.
//
//                The tool returns a list of repositories in a similar to the following JSON example:
//                [
//                    {
//                        "url":"https://github.com/bswanson58/NoiseMusicSystem",
//                        "visibility":"PUBLIC",
//                        "language":"C#",
//                        "createdAt":1372698378.000000000,
//                        "updatedAt":1676375106.000000000,
//                        "pushedAt":1697816613.000000000
//                    }
//                ]
//
//                Summarize the results in a table format similar to the example summary shown below:
//                    Here’s a quick summary of the GitHub repositories you listed, based on their metadata:
//
//                    RepositoryUrl	        Language	Last Updated	Visibility	Description (if available)
//                    ---------------------	--------	------------	----------	-----------------------
//
//                Include Key observations like those show here:
//                    Key observations
//                        •	Most of these are small “hello world” style or workshop/demo repositories.
//                        •	Languages covered include Python, Java, C#, and JavaScript—matching your multi-language development focus.
//                        •	SystemPrereqs and vector-db-samples are the most recently updated/pushed (well into 2025), suggesting they may be currently active or under ongoing maintenance.
//
//                """;

}
