﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.MLAgents;
using System.IO;
using System.Linq;
using UnityEditor.Build.Content;
using System.Reflection;

public class Monitor_Training : MonoBehaviour
{
    private int objectCount = 0;  // Interaction Object의 수
    private TextMesh targetText; // Interaction Object ID의 Text
    public int totalObject = 5; // TODO: Scene에 배치되어 있는 interaction obstacle 개수로 초기화 시키기

    [HideInInspector]
    public int frameCount = 0;
    private List<GoalAndSpawn> goalAreas;
    public float timeScale = 1f;
    [Header("Environment")]
    [Tooltip("Number of Agents [2,n]")]
    public int numOfAgents;
    [Tooltip("Spawn agents gradually instead of all together. If enabled, disable GridSpawn.")]
    public bool spawnGradually;
    [Tooltip("Number of agents to spawn each time [1,10]")]
    public int spawnGraduallyNumber;
    [Tooltip("Spread agents to spawn points equally")]
    public bool spawnEqually;
    [Tooltip("Spawn agents in a grid pattern")]
    public bool gridSpawn;
    [Tooltip("Set all the objects in scene as interaction")]
    public bool setAllInteraction;
    [Tooltip("Seconds to inherit weights by the time agent enter weights zone [0,n]")]
    public float timeToInheritWeights;
    [Tooltip("Max distance can happen in the environment (0,n], used of distance normalization")]
    public float maxDistance;
    [Tooltip("Max distance to target object can happen in the environment (0,n], used of distance normalization")]
    public float maxDistanceToTargetObject;
    [Tooltip("If enabled, agents keep inherited weights when outside weights zone")]
    public bool keepInheritWeights;
    [Tooltip("Radius of weights area (0,n]")]
    public float inheritWeightsDistance;
    //------------------------------
    [Header("Agents")]
    public GameObject agentPrefab;
    [Tooltip("Color agents based on the weights")]
    public bool coloredWeights;
    private GameObject agentParent;
    //------------------------------
    [Tooltip("Run only one episode")]
    public bool oneEpisodeOnly;
    [Tooltip("Unlimited Episode Steps")]
    public bool unlimitedEpisodeSteps;
    [Tooltip("Destroy agent object on goal and recreate")]
    public bool disappearOnGoal;
    //------------------------------
    [Header("Hall/Cross Environment Only")]
    [Tooltip("Set goal point only opposite of spawn direction")]
    public bool oppositeGoal;
    //------------------------------
    [Header("Circular Environment Only")]
    [Tooltip("Spawn agents in a circular pattern in radius circularSpawnRadius")]
    public bool circularSpawn;
    [Tooltip("Radius for circular spawn (0,n]")]
    public float circularSpawnRadius;
    //------------------------------
    [Header("Museum Environment Only")]
    [Tooltip("If selected start agents with goal behaviour until enter weights area.")]
    public bool demoScenes;
    [Tooltip("Use many behaviours simultaneously")]
    public bool multiBehaviors;
    [Tooltip("Percentage of the agents have goal behaviour [0,100], all behaviours must add to 100")]
    public float goalPercentage;
    [Tooltip("Percentage of the agents have group behaviour [0,100], all behaviours must add to 100")]
    public float groupPercentage;
    [Tooltip("Percentage of the agents have interaction behaviour [0,100], all behaviours must add to 100")]
    public float interactionPercentage;
    [Tooltip("Percentage of the agents have avoidance behaviour [0,100], all behaviours must add to 100")]
    public float avoidancePercentage;
    //------------------------------
    [Header("Weights")]
    [Tooltip("If enabled, you can use weights outside the limits by adjusting the min/max too.")]
    public bool extreameWeights;
    public float goalWeight;
    public float collisionWeight;
    public float interactWeight;
    public float avoidanceWeight;
    public float groupWeight;
    //------------------------------
    [Header("Min/Max Weights")]
    public float goalMin = 0.1f;
    public float goalMax = 1.8f;

    public float collMin = 0.5f;
    public float collMax = 3.5f;

    public float interMin = -5.0f;
    public float interMax = 5.0f;

    public float avoidMin = -5.0f;
    public float avoidMax = 5.0f;

    public float groupMin = -3.0f;
    public float groupMax = 5.0f;
    //-------------------------------
    [Header("Phase")]
    public int phase;
    [Tooltip("Seconds to change phase [1,n]")]
    public float changePhaseInterval;
    [HideInInspector] public float scaling;
    //-------------------------------
    [Header("Reward Thresholds")]
    [Tooltip("Distance to set goal arrival [1,n]")]
    public float goalDistanceThreshold;
    [Tooltip("Distance to assume grouping [1,n]")]
    public float groupDistanceThreshold;
    [Tooltip("Distance to assume interaction [1,n]")]
    public float interactionDistanceThreshold;
    [Tooltip("Distance to assume avoidance [1,n]")]
    public float avoidanceDistanceThreshold;
    [Tooltip("Max neighbours to enable group and interaction. [2,n]")]
    public int maxNeighbours;
    //-------------------------------
    Toggle randomToggle;
    Slider goalSlider;
    Slider collSlider;
    Slider groupSlider;
    Slider interSlider;
    Slider avoidSlider;
    //-------------------------------
    List<GameObject> interactionObjects;

    List<GameObject> avoidanceObjects;

    Color obstalceColor;
    //-------------------------------
    private StatsRecorder statsRecorder;
    //-------------------------------
    [Header("Save Routes")]
    //Enable "saveRoutes" if you want to save the route of the agents at current run
    [Tooltip("Enable BEFORE RUN to save routes")]
    public bool saveRoutes;
    //While the scene is playing, enable "stopSaving" to save the routes until that time to .csv files
    [Tooltip("Enable DURING RUNTIME to stop recording and save files")]
    public bool stopSaving = false;
    //The time interval between two captured points, e.g. 0.04
    [Tooltip("Seconds between recorded points")]
    public float timestepInterval;
    //The name f the folder to store the files
    [Tooltip("Save directory")]
    public string directoryPath;
    //------------------------------
    private bool interactionSet = false;
    private bool environmentExportSet = false;

    //Color Picker
    private bool isColorReady = false;
    private List<GameObject> rPicker;
    private List<GameObject> gPicker;
    private List<GameObject> bPicker;
    private List<GameObject> aPicker;
    // TODO: 여기에 avoidance 관련해서 추가해야 할 듯


    public class GoalAndSpawn
    {
        public Collider goalCollider;
        public List<Collider> spawnCollider;
    }

    private void Awake()
    {
        this.randomToggle = GameObject.Find("randomToggle").GetComponent<Toggle>();
        this.randomToggle.onValueChanged.AddListener(delegate { ToggleValueChanged(this.randomToggle); });
        Time.timeScale = this.timeScale;
        targetText = GetComponentInChildren<TextMesh>();
    }

    // Start is called before the first frame update
    void Start()
    {
        //Create directory to save .csv files, if enabled
        if (this.saveRoutes)
        {
            Directory.CreateDirectory(this.directoryPath);
            Directory.CreateDirectory(this.directoryPath + "EnvironmentInfo/");
        }
        this.scaling = transform.localScale.x;
        this.statsRecorder = Academy.Instance.StatsRecorder;
        this.obstalceColor = GameObject.FindGameObjectWithTag("Obstacle").GetComponent<Renderer>().material.color;
        this.phase = 1;
        this.agentParent = GameObject.Find("Agents").gameObject;
        //Find all goal and spawn areas in scene
        this.goalAreas = new List<GoalAndSpawn>();
        initializeGoalAreas();
        //Find all interaction objects, GameObject under "Obstacles"
        this.interactionObjects = getInteractionObjects();

        //Find all avoidance objects, GameObject under "Obstacles"
        this.avoidanceObjects = getAvoidanceObjects();


        //assignIDsToObjects(); //TODO: 진영이형이 추가한 것인데, 이게 뭔지 알아봐야함

        // Create Agents
        StartCoroutine(instantiateAgents());
        if (this.demoScenes == true)
        {
            setWeights(this.collisionWeight, this.goalWeight, this.groupWeight, this.interactWeight, this.avoidanceWeight, false);
        }
        else
            setWeights(0, 0, 0, 0, 0, true);
        setSliderValues(true);
        setInteractionObjects();
        setAvoidanceObjects();

        updateText();
        //Change interaction objects and ewards every "changePhaseInterval" seconds
        InvokeRepeating("newPhase", changePhaseInterval, changePhaseInterval);

    }

    private void assignIDsToObjects()
    {
        // 사용자 정의 ID 할당
        for (int i = 0; i < interactionObjects.Count; i++)
        {
            GameObject obj = interactionObjects[i];
            ObjectIdentifier idComponent = obj.GetComponent<ObjectIdentifier>();
            if (idComponent == null)
            {
                idComponent = obj.AddComponent<ObjectIdentifier>();
            }
            idComponent.objectID = i + 1; // ID는 1부터 시작

            // TextMesh 업데이트
            TextMesh textMesh = obj.GetComponentInChildren<TextMesh>();
            if (textMesh != null)
            {
                textMesh.text = "" + idComponent.objectID;
                textMesh.color = Color.white;
            }
        }
    }

    //Create agents from "AgentPrefab" prefab
    private IEnumerator instantiateAgents()
    {
        if (this.multiBehaviors == true)
        {
            WaitForSeconds wait = new WaitForSeconds(UnityEngine.Random.Range(3, 6));
            List<int> idList = Enumerable.Range(1, this.numOfAgents).ToList();

            while(idList.Count > 0)
            {
                int randomIndex = UnityEngine.Random.Range(0, idList.Count);
                int tempID = idList[randomIndex];
                idList.RemoveAt(randomIndex);
                GameObject tempAgent = Instantiate(this.agentPrefab, new Vector3(0, tempID, 0), Quaternion.identity, this.agentParent.transform);
                if (this.unlimitedEpisodeSteps)
                    tempAgent.GetComponent<Agent_Training>().MaxStep = 0;
                if (this.spawnGradually)
                    if (tempID % this.spawnGraduallyNumber == 0)
                        yield return wait;
            }
        }
        else
        {
            WaitForSeconds wait = new WaitForSeconds(UnityEngine.Random.Range(3, 6));
            for (int i = 1; i <= this.numOfAgents; i++)
            {
                GameObject tempAgent = Instantiate(this.agentPrefab, new Vector3(0, i, 0), Quaternion.identity, this.agentParent.transform);
                if (this.unlimitedEpisodeSteps)
                    tempAgent.GetComponent<Agent_Training>().MaxStep = 0;
                if (this.spawnGradually)
                    if (i % this.spawnGraduallyNumber == 0)
                        yield return wait;
            }
        }
    }

    void ToggleValueChanged(Toggle change)
    {
        if (this.randomToggle.isOn)
        {
            setWeights(0, 0, 0, 0, 0, true);
            updateText();
            setSliderValues(false);
        }
    }
    
    //Update values fo sliders
    private void setSliderValues(bool initial) //TODO: UI 부분 수정해야함
    {
        if (initial)
        {
            this.goalSlider = GameObject.Find("GoalSlider").GetComponent<Slider>();
            this.collSlider = GameObject.Find("CollisionSlider").GetComponent<Slider>();
            this.interSlider = GameObject.Find("InteractSlider").GetComponent<Slider>();
            this.groupSlider = GameObject.Find("GroupSlider").GetComponent<Slider>();

            this.avoidSlider = GameObject.Find("AvoidSlider").GetComponent<Slider>();

            if (this.extreameWeights)
            {
                this.goalSlider.minValue = -4f * (this.goalMax - this.goalMin);
                this.goalSlider.maxValue = 4f * (this.goalMax - this.goalMin);
                this.collSlider.minValue = -4f * (this.collMax - this.collMin);
                this.collSlider.maxValue = 4f * (this.collMax - this.collMin);
                this.interSlider.minValue = -4f * (this.interMax - this.interMin);
                this.interSlider.maxValue = 4f * (this.interMax - this.interMin);
                this.groupSlider.minValue = -4f * (this.groupMax - this.groupMin);
                this.groupSlider.maxValue = 4f * (this.groupMax - this.groupMin);

                this.avoidSlider.minValue = -4f * (this.avoidMax - this.avoidMin);
                this.avoidSlider.maxValue = 4f * (this.avoidMax - this.avoidMin);
            }
            else
            {
                this.goalSlider.minValue = this.goalMin;
                this.goalSlider.maxValue = this.goalMax;
                this.collSlider.minValue = this.collMin;
                this.collSlider.maxValue = this.collMax;
                this.interSlider.minValue = this.interMin;
                this.interSlider.maxValue = this.interMax;
                this.groupSlider.minValue = this.groupMin;
                this.groupSlider.maxValue = this.groupMax;

                this.avoidSlider.minValue = this.avoidMin;
                this.avoidSlider.maxValue = this.avoidMax;
            }
        }

        this.goalSlider.value = this.goalWeight;
        this.collSlider.value = this.collisionWeight;
        this.interSlider.value = this.interactWeight;
        this.groupSlider.value = this.groupWeight;

        this.avoidSlider.value = this.avoidanceWeight;
    }

    private void setRGBA() //TODO: UI 부분 수정해야함
    {
        if (this.isColorReady)
        {
            int red = ((int)(normalizeInRange(this.goalWeight, this.goalMin, this.goalMax) * 255f));
            int green = ((int)(normalizeInRange(this.interactWeight, this.interMin, this.interMax) * 255f));
            int blue = ((int)(normalizeInRange(this.groupWeight, this.groupMin, this.groupMax) * 255f));
            int alpha = ((int)(normalizeInRange(this.collisionWeight, this.collMin, this.collMax) * 255f));

            this.rPicker[0].GetComponent<InputField>().text = red.ToString();
            this.gPicker[0].GetComponent<InputField>().text = green.ToString();
            this.bPicker[0].GetComponent<InputField>().text = blue.ToString();
            this.aPicker[0].GetComponent<InputField>().text = alpha.ToString();

            this.rPicker[1].GetComponent<InputField>().text = (red / 255f).ToString("F2");
            this.gPicker[1].GetComponent<InputField>().text = (green / 255f).ToString("F2");
            this.bPicker[1].GetComponent<InputField>().text = (blue / 255f).ToString("F2");
            this.aPicker[1].GetComponent<InputField>().text = (alpha / 255f).ToString("F2");
        }
        else
        {
            this.rPicker = new List<GameObject>(2);
            this.gPicker = new List<GameObject>(2);
            this.bPicker = new List<GameObject>(2);
            this.aPicker = new List<GameObject>(2);
            this.rPicker.Add(GameObject.Find("InputFieldRed"));
            this.gPicker.Add(GameObject.Find("InputFieldGreen"));
            this.bPicker.Add(GameObject.Find("InputFieldBlue"));
            this.aPicker.Add(GameObject.Find("InputFieldAlpha"));
            this.rPicker.Add(GameObject.Find("InputFieldValueRed"));
            this.gPicker.Add(GameObject.Find("InputFieldValueGreen"));
            this.bPicker.Add(GameObject.Find("InputFieldValueBlue"));
            this.aPicker.Add(GameObject.Find("InputFieldValueAlpha"));
            ColorPicker.Create(Color.white, "", null, null, true);
            this.isColorReady = true;
        }
    }

    private void setGraphPoint() //TODO: UI 부분 수정해야함
    {
        Image bg = GameObject.Find("SquareGraphArea").GetComponent<Image>();
        RectTransform canvas = GameObject.Find("SquareGraphArea").GetComponent<RectTransform>();
        RectTransform pointer = GameObject.Find("GraphPointer").GetComponent<RectTransform>();

        float nGoal = normalizeInRange(this.goalWeight, this.goalMin, this.goalMax);
        float nInteract = normalizeInRange(this.interactWeight, this.interMin, this.interMax);
        float nGroup = normalizeInRange(this.groupWeight, this.groupMin, this.groupMax);
        float nColl = normalizeInRange(this.collisionWeight, this.collMin, this.collMax);

        GameObject.Find("goalGraphText").GetComponent<Text>().fontSize = (int) Mathf.Clamp(20f * nGoal, 4, 20);
        GameObject.Find("interactGraphText").GetComponent<Text>().fontSize = (int)Mathf.Clamp(20f * nInteract, 4, 20);
        GameObject.Find("groupGraphText").GetComponent<Text>().fontSize = (int)Mathf.Clamp(20f * nGroup, 4, 20);
        GameObject.Find("collisionGraphText").GetComponent<Text>().fontSize = (int)Mathf.Clamp(20f * nColl, 4, 20);

        Vector2 center = new Vector3(0, 0);

        Vector2 goal = new Vector2(-1f, 1f) * nGoal;
        Vector2 collision = new Vector2(1f, 1f) * nColl;
        Vector2 interact = new Vector2(1f, -1f) * nInteract;
        Vector2 group = new Vector2(-1f, -1f) * nGroup;
        center = (goal + collision + interact + group) / 4;

        float pointerPosX = center.x * (canvas.rect.width - 6f);
        float pointerPosY = center.y * (canvas.rect.height - 6f);

        pointer.anchoredPosition = new Vector2(pointerPosX, pointerPosY);

        Color color = new Color(nGoal, nInteract, nGroup, nColl);
        bg.color = color;

    }

    private void Update()
    {
        if (!this.randomToggle.isOn)
        {
            setWeights(this.collSlider.value, this.goalSlider.value, this.groupSlider.value, this.interSlider.value, this.avoidSlider.value, false);
            updateText();
        }
        setRGBA();
        setGraphPoint();
        if (this.saveRoutes == true && this.interactionSet == true && this.environmentExportSet == false)
            saveEnvironmentInfo();
    }

    //New Phase, set new random weights, if random is enabled
    private void newPhase()
    {
        this.phase++;
        if (this.randomToggle.isOn)
            setWeights(0, 0, 0, 0, 0, true);
        setInteractionObjects();
        setAvoidanceObjects();
        updateText();
        setSliderValues(false);
    }

    //Update Weights, if random equals to true, use random, else use function inputs as weights
    private void setWeights(float collision, float goal, float groub, float interact, float avoid, bool random)
    {
        if (random)
        {
            this.collisionWeight = Random.Range((this.collMin + this.collMax)/2.5f, this.collMax/1.25f);
            this.goalWeight = Random.Range(this.goalMin, this.goalMax);
            this.groupWeight = Random.Range(this.groupMin, this.groupMax);
            this.interactWeight = Random.Range(this.interMin, this.interMax);

            this.avoidanceWeight = Random.Range(this.avoidMin, this.avoidMax);

            //this.goalWeight = Random.Range(0.0F, 1.0F) < 0.5F ? this.goalMin * Random.Range(0.5F, 1.0F) : this.goalMax * Random.Range(0.5F, 1.0F);
            //this.groupWeight = Random.Range(0.0F, 1.0F) < 0.5F ? this.groupMin * Random.Range(0.5F, 1.0F) : this.groupMax * Random.Range(0.5F, 1.0F);
            //this.interactWeight = Random.Range(0.0F, 1.0F) < 0.5F ? this.interMin * Random.Range(0.5F, 1.0F) : this.interMax * Random.Range(0.5F, 1.0F);
        }
        else
        {
            this.collisionWeight = collision;
            this.goalWeight = goal;
            this.groupWeight = groub;
            this.interactWeight = interact;
            this.avoidanceWeight = avoid;
        }
        statsRecorder.Add("Goal Weight", this.goalWeight);
        statsRecorder.Add("Collide Weight", this.collisionWeight);
        statsRecorder.Add("Group Weight", this.groupWeight);
        statsRecorder.Add("Interact Weight", this.interactWeight);
        statsRecorder.Add("Avoidance Weight", this.avoidanceWeight);
    }

    //Update weight text
    private void updateText() //TODO: UI 부분 수정해야함
    {
        GameObject.Find("phaseText").GetComponent<Text>().text = this.phase.ToString();
        GameObject.Find("GoalW").GetComponent<Text>().text = normalizeInRange(this.goalWeight, this.goalMin, this.goalMax).ToString("0.0");
        GameObject.Find("CollisionW").GetComponent<Text>().text = normalizeInRange(this.collisionWeight, this.collMin, this.collMax).ToString("0.0");
        GameObject.Find("InteractW").GetComponent<Text>().text = normalizeInRange(this.interactWeight, this.interMin, this.interMax).ToString("0.0");
        GameObject.Find("GroupW").GetComponent<Text>().text = normalizeInRange(this.groupWeight, this.groupMin, this.groupMax).ToString("0.0");
    }


    //Return list of goal and spawn areas
    public List<GoalAndSpawn> getGoalAreas()
    {
        return this.goalAreas;
    }

    public float normalizeInRange(float value, float min, float max)
    {
        float scaledValue = (value - min) / (max - min);
        return scaledValue;
    }

    //Return list of interaction objects
    private List<GameObject> getInteractionObjects()
    {
        List<GameObject> obj = new List<GameObject>();
        Transform parent = GameObject.Find("Obstacles").transform;
        if (parent.GetChild(0).gameObject.name.Contains("Parent"))
        {
            foreach (Transform child in parent)
            {
                foreach (Transform child2 in child)
                    obj.Add(child2.gameObject);
            }
        }
        else
        {
            foreach (Transform child in parent)
            {
                obj.Add(child.gameObject);
            }
        }   
        return obj;
    }

    //At every phase initialize a subset of objects as interaction
    private void setInteractionObjects()
    {
        //// <Old Version>
        //if (this.setAllInteraction)
        //{
        //    foreach (GameObject child in this.interactionObjects)
        //    {
        //        child.tag = "Interaction";
        //        child.GetComponent<Renderer>().material.color = Color.red;
        //    }
        //    this.interactionSet = true;
        //    return;
        //}
        //foreach (GameObject child in this.interactionObjects)
        //{
        //    child.tag = "Obstacle";
        //    child.GetComponent<Renderer>().material.color = this.obstalceColor;
        //}
        //int len = this.interactionObjects.Count;
        //int times = Random.Range(3, len);
        //for (int i = 0; i < times; i++)
        //{
        //    int index = Random.Range(0, len);
        //    this.interactionObjects[i].tag = "Interaction";
        //    this.interactionObjects[i].GetComponent<Renderer>().material.color = Color.red;
        //}
        //this.interactionSet = true;

        ////////////////////////////////////////////////////////////////////////////////////
        // <New Version>

        // 모든 객체를 초기화
        foreach (GameObject child in this.interactionObjects)
        {
            child.tag = "Obstacle"; // Obstacle로 태그 초기화
            child.GetComponent<Renderer>().material.color = this.obstalceColor; // 기본 색상

            // ObjectIdentifier 확인 및 없으면 추가
            ObjectIdentifier idComponent = child.GetComponent<ObjectIdentifier>();
            if (idComponent == null)
            {
                idComponent = child.AddComponent<ObjectIdentifier>();
            }

            // ID 설정 (ID가 이미 있으면 유지)
            if (idComponent.objectID <= 0)
            {
                idComponent.objectID = this.interactionObjects.IndexOf(child) + 1; // 1부터 시작하는 ID
            }

            // TextMesh 업데이트
            TextMesh textMesh = child.GetComponentInChildren<TextMesh>();
            if (textMesh != null)
            {
                textMesh.text = "" + idComponent.objectID; // Obstacle ID 표시
                textMesh.color = Color.white;
            }
        }

        // 랜덤으로 Interaction 객체 설정
        int len = this.interactionObjects.Count;
        int interactionCount = Random.Range(3, len + 1); // 최소 1개 이상의 Interaction 설정
        List<int> selectedIndices = new List<int>();

        for (int i = 0; i < interactionCount; i++)
        {
            int index;
            do
            {
                index = Random.Range(0, len);
            } while (selectedIndices.Contains(index)); // 중복 방지
            selectedIndices.Add(index);

            GameObject obj = this.interactionObjects[index];
            obj.tag = "Interaction"; // Interaction으로 태그 변경
            obj.GetComponent<Renderer>().material.color = Color.red; // 색상 변경

            // TextMesh 업데이트
            TextMesh textMesh = obj.GetComponentInChildren<TextMesh>();
            if (textMesh != null)
            {
                ObjectIdentifier idComponent = obj.GetComponent<ObjectIdentifier>();
                textMesh.text = "" + idComponent.objectID; // Interaction ID 표시 (Obstacle ID와 동일)
                textMesh.color = Color.green; // Interaction 상태 색상
            }
        }

        // Interaction 객체 수를 objectCount로 설정
        this.objectCount = interactionCount; // TODO: objectCount가 사용되지 않고 있는 것 같음. 이 부분에 대해서 알아봐야 함
        this.interactionSet = true;

        Debug.Log($"Interaction Objects Count: {this.objectCount}");
    }

    //Return list of avoidance objects
    private List<GameObject> getAvoidanceObjects()
    {
        List<GameObject> obj = new List<GameObject>();
        Transform parent = GameObject.Find("Avoid").transform;
        if (parent.GetChild(0).gameObject.name.Contains("Parent"))
        {
            foreach (Transform child in parent)
            {
                foreach (Transform child2 in child)
                    obj.Add(child2.gameObject);
            }
        }
        else
        {
            foreach (Transform child in parent)
            {
                obj.Add(child.gameObject);
            }
        }
        return obj;
    }

    //
    private void setAvoidanceObjects()
    {
        foreach (GameObject child in this.avoidanceObjects)
        {
            child.tag = "Avoid";
            child.GetComponent<Renderer>().material.color = Color.white;

            // TextMesh 업데이트
            TextMesh textMesh = child.GetComponentInChildren<TextMesh>();
            if (textMesh != null)
            {
                textMesh.text = "Avoid";
                textMesh.color = Color.black;
            }
        }
    }

    //Find goal and spawn areas in scene
    private void initializeGoalAreas()
    {
        //Initialize inside goal areas
        GameObject[] goalAreasParent = GameObject.FindGameObjectsWithTag("GoalArea");
        foreach (GameObject area in goalAreasParent)
        {
            GoalAndSpawn temp = new GoalAndSpawn();
            temp.goalCollider = area.GetComponent<Collider>();
            temp.spawnCollider = new List<Collider>();
            for (int j = 0; j < area.transform.childCount; j++)
                temp.spawnCollider.Add(area.transform.GetChild(j).GetComponent<Collider>());
            this.goalAreas.Add(temp);
        }
    }

    private void FixedUpdate()
    {
        this.frameCount++;
    }

    private void saveEnvironmentInfo()
    {
        this.environmentExportSet = true;
        string filePath = this.directoryPath + "EnvironmentInfo/enviInfo.csv";
        StreamWriter writer = new StreamWriter(filePath);

        writer.WriteLine(this.maxDistance);
        writer.WriteLine(this.groupDistanceThreshold);
        writer.WriteLine(this.interactionDistanceThreshold);
        writer.WriteLine(this.avoidanceDistanceThreshold);
        writer.WriteLine(this.maxNeighbours);

        GameObject[] interObjects = GameObject.FindGameObjectsWithTag("Interaction"); // TODO: 이 부분 avoidance object에도 적용하기
        foreach(GameObject g in interObjects)
        {
            Collider c = g.GetComponent<Collider>();
            Vector3 center = c.bounds.center;
            float minX = c.bounds.min.x;
            float maxX = c.bounds.max.x;
            float minZ = c.bounds.min.z;
            float maxZ = c.bounds.max.z;
            writer.WriteLine(center.x.ToString("F4") + ";" + center.z.ToString("F4") + ";" + minX.ToString("F4") + ";" + maxX.ToString("F4") + ";" + minZ.ToString("F4") + ";" + maxZ.ToString("F4"));
        }

        writer.Flush();
        writer.Close();
    }

    //Save routes of every agent to .csv, if enabled
    public void saveRoute(Vector3 start, Vector3 end, int agentCollisions, string name, float reward, List<float[]> list)
    {
        float threshold = -100;
        if (reward > threshold && list.Count > 5)
        {
            string filePath = this.directoryPath + name + ".csv";
            StreamWriter writer = new StreamWriter(filePath);

            int step = (int)(this.timestepInterval / Time.fixedDeltaTime);

            writer.WriteLine(start.x.ToString("F4") + ";" + start.z.ToString("F4") + ";" + end.z.ToString("F4") + ";" + end.z.ToString("F4") + ";" + agentCollisions);

            for (int i = 1; i < list.Count; i += step)
            {
                writer.WriteLine(list[i][0].ToString("F4") + ";" + list[i][1].ToString("F4") + ";" + list[i][2].ToString("F4") + ";" + list[i][3].ToString("F4") + ";" +
                    list[i][4].ToString("F4") + ";" + list[i][5].ToString("F4") + ";" + list[i][6].ToString("F4") + ";" + list[i][7].ToString("F4") + ";" + list[i][8].ToString("F4") + ";" + list[i][9].ToString("F4"));
            }
            writer.Flush();
            writer.Close();
        }
    }
}


