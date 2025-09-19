// Application constants

export const DEFAULT_CONTEXT_TYPE = 'github' as const;

export const GITHUB_DEFAULT_CONTEXT = `
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

Only return the JSON, no other text.
`;