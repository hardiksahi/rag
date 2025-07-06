import os
import dotenv
from typing import Any, Dict
import numpy as np
from langchain_neo4j import Neo4jGraph

dotenv.load_dotenv()


def _get_current_hospitals():
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        sanitize=False,  ## To avoid getting embedding properties, set sanitize as True
    )

    current_hospitals = graph.query(
        """
        MATCH (h:Hospital)
        RETURN h.name AS hospital_name
    """
    )
    return [d["hospital_name"].lower() for d in current_hospitals]


def _get_current_wait_time_minutes(hospital: str) -> int:
    current_hospitals = _get_current_hospitals()

    if hospital.lower() not in current_hospitals:
        return -1

    return np.random.randint(low=0, high=600)


def get_current_wait_times(hospital: str) -> str:
    wait_time_in_min = _get_current_wait_time_minutes(hospital)

    if wait_time_in_min == -1:
        return f"Hospital {hospital} does not exist"

    hours, minutes = divmod(wait_time_in_min, 60)

    if hours > 0:
        return f"{hours} hours {minutes} minutes"
    else:
        return f"{minutes} minutes"


def get_most_available_hospital(_: Any) -> Dict[str, float]:  ## Throwaway input
    current_hospitals = _get_current_hospitals()

    current_wait_times = [_get_current_wait_time_minutes(h) for h in current_hospitals]
    best_time_idx = np.argmin(current_wait_times)
    best_hospital = current_hospitals[best_time_idx]
    best_wait_time = current_wait_times[best_time_idx]

    return {best_hospital: best_wait_time}


if __name__ == "__main__":
    print(
        f"Wait time at Wallace-Hamilton: {get_current_wait_times('Wallace-Hamilton')}"
    )

    print(f"Wait time at Fake hospital: {get_current_wait_times('Fake hospital')}")

    print(f"Earliest available hospital: {get_most_available_hospital(None)}")
