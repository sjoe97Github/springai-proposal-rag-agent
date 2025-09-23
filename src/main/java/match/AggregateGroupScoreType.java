package match;

public enum AggregateGroupScoreType {
    SUM("sum"),
    AVG("avg"),
    MAX("max"),
    SOFTMAX("softmax");

    private final String type;

    AggregateGroupScoreType(String type) {
        this.type = type;
    }

    public String getType() {
        return type;
    }

    public static AggregateGroupScoreType fromString(String type) {
        for (AggregateGroupScoreType t : values()) {
            if (t.type.equalsIgnoreCase(type)) {
                return t;
            }
        }
        return null;
    }
}

