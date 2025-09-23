package match;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class AggregateScoringAlgorithm {
    private String aggregateGroupsScoreType;

    public AggregateScoringAlgorithm(@Value("${app.match.aggregate-group-score}") String defaultAggregateGroupsScoreType) {
        this.aggregateGroupsScoreType = defaultAggregateGroupsScoreType;
    }

    public String getScoringAlgorithm() {
        return aggregateGroupsScoreType;
    }

    public void setScoringAlgorithm(String systemContext) {
        this.aggregateGroupsScoreType = systemContext;
    }
}
