import { Configuration, OpenAIApi } from "openai";
import { PineconeClient } from "@pinecone-database/pinecone";
// Set API Keys
const OPENAI_API_KEY = "";
const PINECONE_API_KEY = "";
const PINECONE_ENVIRONMENT = "us-east4-gcp"; // Pinecone Environment (eg. "us-east1-gcp")

// Set Variables
const YOURTableName = "your-table";
const OBJECTIVE = "solve the homeless problem in a random US city";
const YOUR_firstTask = "Develop a task list.";

// Print OBJECTIVE
console.log("*****OBJECTIVE*****");
console.log(OBJECTIVE);

// Configure OpenAI and Pinecone
const configuration = new Configuration({
  apiKey: OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

const pinecone = new PineconeClient();
await pinecone.init({
  environment: PINECONE_ENVIRONMENT,
  apiKey: PINECONE_API_KEY,
});

// Create Pinecone index
const tableName = YOURTableName;
const dimension = 1536;
const metric = "cosine";
const podType = "p1.x1";
const pineconeIndexList = await pinecone.listIndexes();

if (!pineconeIndexList.includes(tableName)) {
  await pinecone.createIndex({
    createRequest: { name: tableName, dimension, metric, pod_type: podType },
  });
}

// Connect to the index
const index = await pinecone.Index(tableName);

// Task list
const taskList = [];

async function getAdaEmbedding(text) {
  text = text.replace(/\n/g, " ");
  const embeddingResponse = await openai.createEmbedding({
    input: [text],
    model: "text-embedding-ada-002",
  });
  return embeddingResponse.data.data[0].embedding;
}

async function taskCreationAgent(objective, result, taskDescription, taskList) {
  const prompt = `You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: ${objective}, The last completed task has the result: ${result}. This result was based on this task description: ${taskDescription}. These are incomplete tasks: ${taskList.join(
    ", "
  )}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array.`;
  const response = await openai.createCompletion({
    model: "text-davinci-003",
    prompt,
    temperature: 0.5,
    max_tokens: 100,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  });
  const newTasks = response.data.choices[0].text.trim().split("\n");
  return newTasks.map((taskName) => ({ taskName }));
}

async function prioritizationAgent(thisTaskId) {
  let taskNames = taskList.map((t) => t["taskName"]);
  let nextTaskId = parseInt(thisTaskId) + 1;
  let prompt = `You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: ${taskNames}. Consider the ultimate objective of your team:${OBJECTIVE}. Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number ${nextTaskId}.`;
  let response = await openai.createCompletion({
    model: "text-davinci-003",
    prompt,
    temperature: 0.5,
    max_tokens: 1000,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  });
  let newTasks = response.data.choices[0].text.trim().split("\n");
  for (let taskString of newTasks) {
    let taskParts = taskString.trim().split(".", 1);
    if (taskParts.length === 2) {
      let taskId = taskParts[0].trim();
      let taskName = taskParts[1].trim();
      taskList.push({ taskId, taskName });
    }
  }
}

async function executionAgent(objective, task) {
  let context = await contextAgent(objective, YOURTableName, 5);
  let prompt = `You are an AI who performs one task based on the following objective: ${objective}.\nTake into account these previously completed tasks: ${context}\nYour task: ${task}\nResponse:`;
  let response = await openai.createCompletion({
    model: "text-davinci-003",
    prompt,
    temperature: 0.7,
    max_tokens: 2000,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  });
  return response.data.choices[0].text.trim();
}

async function contextAgent(query, index, n) {
  const queryEmbedding = await getAdaEmbedding(query);
  const indexClient = await pinecone.Index(index);
  const results = await indexClient.query({
      queryRequest: {
        vector: queryEmbedding,
        topK: n,
        includeMetadata: true,
      },
    });

  const sortedResults = results.matches.sort((a, b) => b.score - a.score);
  return sortedResults.map((item) => item.metadata.task);
}

// Add the first task
const firstTask = {
  taskId: 1,
  taskName: YOUR_firstTask,
};

taskList.push(firstTask);

// Main loop
let taskIDCounter = 1;

// this has the power to stop it from taking over the world after certain amount of work
// set to true and it can work for you forever tho :)
let yeee = 10; 
while (yeee) {
  yeee--;

  if (taskList.length > 0) {
    // Print the task list
    console.log(`*****TASK LIST*****`);
    for (let t of taskList) {
      console.log(`${t.taskId}: ${t.taskName}`);
    }

    // Step 1: Pull the first task
    let task = taskList.shift();
    console.log(`*****NEXT TASK*****`);
    console.log(`${task.taskId}: ${task.taskName}`);

    // Send to execution function to complete the task based on the context
    let result = await executionAgent(OBJECTIVE, task.taskName);
    let thisTaskID = parseInt(task.taskId);
    console.log(`*****TASK RESULT****`);
    console.log(result);

    // Step 2: Enrich result and store in Pinecone
    let enrichedResult = { data: result }; // This is where you should enrich the result if needed
    let resultID = `result_${task.taskId}`;
    let vector = enrichedResult.data; // extract the actual result from the dictionary
    const embedding = await getAdaEmbedding(vector);

    index.upsert({
      upsertRequest: {
        vectors: [
          {
            id: resultID,
            values: embedding,
            metadata: { task: task.taskName, result: result },
          },
        ],
      },
    });

    // Step 3: Create new tasks and reprioritize task list
    let newTasks = await taskCreationAgent(
      OBJECTIVE,
      enrichedResult,
      task.taskName,
      taskList.map((t) => t.taskName)
    );

    for (let newTask of newTasks) {
      taskIDCounter += 1;
      newTask.taskId = taskIDCounter;
      taskList.push(newTask);
    }
    await prioritizationAgent(thisTaskID);
  }
  // Sleep before checking the task list again
  await new Promise((resolve) => setTimeout(resolve, 1000));
}
