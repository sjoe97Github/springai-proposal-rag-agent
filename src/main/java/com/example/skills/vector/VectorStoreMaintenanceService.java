package com.example.skills.vector;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

@Service
public class VectorStoreMaintenanceService {
    private final JdbcTemplate jdbcTemplate;
    // Change this to your actual vector table name if different
    private static final String VECTOR_TABLE = "vector_store";

    @Autowired
    public VectorStoreMaintenanceService(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public void clearPgVectorTable() {
        jdbcTemplate.execute("TRUNCATE TABLE " + VECTOR_TABLE + " RESTART IDENTITY CASCADE");
    }

    public int countVectors() {
        Integer count = jdbcTemplate.queryForObject("SELECT COUNT(*) FROM " + VECTOR_TABLE, Integer.class);
        return count != null ? count : 0;
    }
}

