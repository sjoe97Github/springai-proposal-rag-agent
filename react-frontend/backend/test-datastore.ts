// 
// TODO - Work in progress - rexamine now that dataStore.ts has been edited
//
import { DataStore } from './dataStore';

const dataStore = new DataStore();

console.log('Testing updated DataStore...');

// Test resume matching with actual query terms from the data
console.log('\n1. Testing Java C# Python query:');
const results1 = dataStore.getResumeMatches('Java C# Python');
console.log(`Found ${results1.length} candidates:`);
results1.forEach(candidate => {
    console.log(`- ${candidate.candidateId}: ${candidate.finalScore} - ${candidate.shortExplanation}`);
});

console.log('\n2. Testing Automation Security query:');
const results2 = dataStore.getResumeMatches('Automation Security');
console.log(`Found ${results2.length} candidates:`);
results2.forEach(candidate => {
    console.log(`- ${candidate.candidateId}: ${candidate.finalScore} - ${candidate.shortExplanation}`);
});

console.log('\n3. Testing TypeScript JavaScript query:');
const results3 = dataStore.getResumeMatches('TypeScript JavaScript');
console.log(`Found ${results3.length} candidates:`);
results3.forEach(candidate => {
    console.log(`- ${candidate.candidateId}: ${candidate.finalScore} - ${candidate.shortExplanation}`);
    if (candidate.github) {
        console.log(`  GitHub: ${candidate.github}`);
    }
    if (candidate.reposList && Array.isArray(candidate.reposList)) {
        console.log(`  Repositories: ${candidate.reposList.length} repos`);
    }
});

console.log('\nDataStore test completed!');