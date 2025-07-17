#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

console.log("🔧 Timbre App Development Setup");
console.log("===============================\n");

// Check if .env.local exists
const envPath = path.join(__dirname, "..", ".env.local");
const envExamplePath = path.join(__dirname, "..", "env.example");

if (!fs.existsSync(envPath)) {
  console.log("❌ .env.local file not found!");
  console.log("📝 Creating .env.local from template...");

  if (fs.existsSync(envExamplePath)) {
    fs.copyFileSync(envExamplePath, envPath);
    console.log("✅ .env.local created from template");
    console.log("⚠️  Please update .env.local with your actual credentials\n");
  } else {
    console.log("❌ env.example not found");
    process.exit(1);
  }
} else {
  console.log("✅ .env.local file exists");
}

// Check required environment variables
const requiredVars = [
  "NEXT_PUBLIC_SUPABASE_URL",
  "NEXT_PUBLIC_SUPABASE_ANON_KEY",
  "SPOTIFY_CLIENT_ID",
  "SPOTIFY_CLIENT_SECRET",
  "LASTFM_API_KEY",
];

console.log("\n🔍 Checking environment variables...");

const envContent = fs.readFileSync(envPath, "utf8");
const missingVars = [];

requiredVars.forEach((varName) => {
  if (!envContent.includes(`${varName}=`)) {
    missingVars.push(varName);
  }
});

if (missingVars.length > 0) {
  console.log("❌ Missing required environment variables:");
  missingVars.forEach((varName) => {
    console.log(`   - ${varName}`);
  });
  console.log("\n📝 Please add these to your .env.local file");
} else {
  console.log("✅ All required environment variables are present");
}

console.log("\n🚀 Next steps:");
console.log("1. Update .env.local with your actual credentials");
console.log("2. Run: npm run dev");
console.log("3. Open: http://localhost:3000");
console.log("\n📚 For setup instructions, see README.md");
