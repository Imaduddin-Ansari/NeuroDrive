"""
International Traffic Rules Knowledge Base Generator
Generates structured JSON that maps traffic signs/scenarios to rules for ML model integration
Based on international traffic conventions and common practices across multiple countries
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class Priority(Enum):
    CRITICAL = "critical"  # Must stop/yield immediately
    HIGH = "high"          # Should yield/slow down
    MEDIUM = "medium"      # Be cautious
    LOW = "low"            # Informational

class ActionType(Enum):
    STOP = "stop"
    YIELD = "yield"
    SLOW_DOWN = "slow_down"
    GIVE_WAY = "give_way"
    CHANGE_LANE = "change_lane"
    PROCEED_WITH_CAUTION = "proceed_with_caution"
    MAINTAIN_DISTANCE = "maintain_distance"
    KEEP_RIGHT = "keep_right"
    KEEP_LEFT = "keep_left"
    NO_ENTRY = "no_entry"
    INFORMATIONAL = "informational"

@dataclass
class TrafficRule:
    rule_id: str
    category: str
    trigger: str
    description: str
    priority: str
    actions: List[str]
    conditions: List[str]
    exceptions: List[str]
    distance_requirements: Dict[str, Any]
    speed_requirements: Dict[str, Any]
    applicable_regions: List[str]  # Countries/regions where this applies
    regional_variations: Optional[Dict[str, str]] = None

class TrafficRulesKnowledgeBase:
    def __init__(self):
        self.rules = []
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize comprehensive international traffic rules"""
        
        # ============ EMERGENCY VEHICLES ============
        self.add_rule(TrafficRule(
            rule_id="EMG_001",
            category="emergency_vehicles",
            trigger="emergency_vehicle_detected",
            description="Give way to emergency vehicles with active sirens or flashing lights",
            priority=Priority.CRITICAL.value,
            actions=[
                ActionType.GIVE_WAY.value,
                ActionType.CHANGE_LANE.value,
                ActionType.SLOW_DOWN.value
            ],
            conditions=[
                "emergency_vehicle_approaching",
                "siren_active OR lights_flashing",
                "ambulance OR police OR fire_truck"
            ],
            exceptions=[
                "if_unsafe_to_move",
                "if_would_enter_intersection_unsafely"
            ],
            distance_requirements={
                "minimum_clearance": "pull_over_to_side",
                "following_distance": "200_meters_minimum",
                "do_not_follow_closely": True
            },
            speed_requirements={
                "action": "reduce_speed_significantly",
                "stop_if_necessary": True
            },
            applicable_regions=["USA", "UK", "EU", "UAE", "Canada", "Australia", "India", "China", "Japan", "Global"],
            regional_variations={
                "USA": "Pull to right side",
                "UK_Australia_India_Japan": "Pull to left side in left-hand traffic countries",
                "Germany": "Create emergency corridor (Rettungsgasse) between lanes on highways"
            }
        ))
        
        # ============ ROUNDABOUTS / TRAFFIC CIRCLES ============
        self.add_rule(TrafficRule(
            rule_id="RND_001",
            category="roundabouts",
            trigger="roundabout_sign_detected",
            description="Vehicles already in the roundabout have right of way",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.YIELD.value,
                ActionType.SLOW_DOWN.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "approaching_roundabout",
                "yield_to_traffic_in_roundabout",
                "safe_gap_available"
            ],
            exceptions=[
                "traffic_signal_overrides_yield"
            ],
            distance_requirements={
                "yield_distance": "sufficient_to_stop_safely",
                "entry_gap": "minimum_4_seconds"
            },
            speed_requirements={
                "approach_speed": "20_to_30_kmh",
                "circulating_speed": "maintain_steady_pace"
            },
            applicable_regions=["UK", "EU", "USA", "UAE", "Australia", "Canada", "Global"],
            regional_variations={
                "UK_Australia_India": "Circulate clockwise, yield to right",
                "USA_EU_UAE": "Circulate counterclockwise, yield to left",
                "France": "Arc de Triomphe has reversed priority"
            }
        ))
        
        # ============ STOP SIGNS ============
        self.add_rule(TrafficRule(
            rule_id="SIGN_001",
            category="traffic_signs",
            trigger="stop_sign_detected",
            description="Come to complete stop, yield to all cross traffic",
            priority=Priority.CRITICAL.value,
            actions=[
                ActionType.STOP.value,
                ActionType.YIELD.value
            ],
            conditions=[
                "must_stop_completely",
                "wheels_must_stop_rotating",
                "check_all_directions",
                "yield_to_all_traffic"
            ],
            exceptions=[
                "none"
            ],
            distance_requirements={
                "stop_position": "at_or_before_stop_line",
                "if_no_line": "before_entering_intersection"
            },
            speed_requirements={
                "final_speed": "0_kmh",
                "stop_duration": "minimum_3_seconds_recommended"
            },
            applicable_regions=["USA", "Canada", "UAE", "Australia", "South Africa", "Global"],
            regional_variations={
                "USA_Canada": "4-way stops: first to arrive goes first, ties go to right",
                "South_Africa": "Red octagon with 'STOP'",
                "Note": "Less common in Europe where yield signs are preferred"
            }
        ))
        
        # ============ YIELD / GIVE WAY SIGNS ============
        self.add_rule(TrafficRule(
            rule_id="SIGN_002",
            category="traffic_signs",
            trigger="yield_sign_detected",
            description="Slow down and prepare to give way to traffic on priority road",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.YIELD.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "traffic_on_priority_road_has_right_of_way",
                "proceed_only_when_safe"
            ],
            exceptions=[
                "no_traffic_on_priority_road"
            ],
            distance_requirements={
                "approach_distance": "reduce_speed_early",
                "safe_gap": "minimum_5_seconds"
            },
            speed_requirements={
                "approach_speed": "reduce_significantly",
                "be_prepared_to_stop": True
            },
            applicable_regions=["USA", "UK", "EU", "UAE", "Canada", "Australia", "Global"],
            regional_variations={
                "USA": "Yellow triangle pointing down or white triangle",
                "UK_EU": "Red triangle pointing down with 'Give Way'",
                "Australia": "Red and white triangle"
            }
        ))
        
        # ============ PEDESTRIAN CROSSINGS ============
        self.add_rule(TrafficRule(
            rule_id="PED_001",
            category="pedestrians",
            trigger="pedestrian_crossing_detected",
            description="Stop and give way to pedestrians at designated crossings",
            priority=Priority.CRITICAL.value,
            actions=[
                ActionType.STOP.value,
                ActionType.GIVE_WAY.value,
                ActionType.MAINTAIN_DISTANCE.value
            ],
            conditions=[
                "pedestrian_on_crossing",
                "pedestrian_waiting_to_cross",
                "zebra_crossing_or_marked_crossing"
            ],
            exceptions=[
                "traffic_signal_controls_crossing"
            ],
            distance_requirements={
                "stop_distance": "before_crossing_line",
                "safety_margin": "5_meters_from_pedestrian"
            },
            speed_requirements={
                "approach_speed": "reduce_to_30_40_kmh",
                "stop_if_pedestrian_present": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "UK": "Zebra crossings with Belisha beacons",
                "USA": "Marked crosswalks, stop for pedestrians in crosswalk",
                "Japan": "Strict pedestrian priority",
                "Germany": "Pedestrians must be on crossing before stopping required"
            }
        ))
        
        # ============ SCHOOL ZONES ============
        self.add_rule(TrafficRule(
            rule_id="ZONE_001",
            category="special_zones",
            trigger="school_zone_sign_detected",
            description="Reduce speed significantly in school zones during active hours",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "school_zone_sign_present",
                "school_hours_may_be_active",
                "children_may_be_present"
            ],
            exceptions=[
                "outside_posted_hours_if_indicated"
            ],
            distance_requirements={
                "increased_awareness_zone": "scan_for_children"
            },
            speed_requirements={
                "maximum_speed": "typically_25_40_kmh",
                "posted_limit_varies_by_country": True
            },
            applicable_regions=["USA", "Canada", "UK", "Australia", "UAE", "EU", "Global"],
            regional_variations={
                "USA": "Usually 15-25 mph (25-40 kmh) during school hours",
                "UK": "20 mph zones common near schools",
                "Australia": "40 kmh school zones with flashing lights"
            }
        ))
        
        # ============ TRAFFIC LIGHTS - RED ============
        self.add_rule(TrafficRule(
            rule_id="LIGHT_001",
            category="traffic_signals",
            trigger="red_traffic_light_detected",
            description="Stop at red traffic light",
            priority=Priority.CRITICAL.value,
            actions=[
                ActionType.STOP.value
            ],
            conditions=[
                "red_light_active",
                "stop_at_stop_line_or_before_intersection"
            ],
            exceptions=[
                "right_turn_on_red_after_stop_if_permitted",
                "left_turn_on_red_in_left_hand_traffic_if_permitted",
                "green_arrow_permits_turn"
            ],
            distance_requirements={
                "stop_position": "at_stop_line"
            },
            speed_requirements={
                "final_speed": "0_kmh"
            },
            applicable_regions=["Global"],
            regional_variations={
                "USA_Canada": "Right turn on red permitted after stop unless posted",
                "UK_Australia_Japan": "No turn on red (left turn in these countries)",
                "EU": "Generally no turn on red unless green arrow present",
                "China": "Right turn permitted unless specifically prohibited"
            }
        ))
        
        # ============ TRAFFIC LIGHTS - YELLOW/AMBER ============
        self.add_rule(TrafficRule(
            rule_id="LIGHT_002",
            category="traffic_signals",
            trigger="yellow_traffic_light_detected",
            description="Prepare to stop, proceed only if too close to stop safely",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "yellow_light_active",
                "determine_point_of_no_return"
            ],
            exceptions=[
                "too_close_to_stop_safely"
            ],
            distance_requirements={
                "evaluate_stopping_distance": "can_stop_safely_before_intersection"
            },
            speed_requirements={
                "reduce_speed": "prepare_to_stop"
            },
            applicable_regions=["Global"],
            regional_variations={
                "USA": "Amber light, stop if safe to do so",
                "UK": "Amber means stop unless unsafe",
                "China": "Yellow often disregarded (enforcement varies)"
            }
        ))
        
        # ============ SPEED LIMITS ============
        self.add_rule(TrafficRule(
            rule_id="SPEED_001",
            category="speed_management",
            trigger="speed_limit_sign_detected",
            description="Adjust speed to posted limit and road conditions",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.INFORMATIONAL.value
            ],
            conditions=[
                "speed_limit_sign_detected",
                "adjust_to_weather_and_traffic",
                "enforce_maximum_speed"
            ],
            exceptions=[],
            distance_requirements={
                "stopping_distance": "adjust_for_speed",
                "following_distance": "2_seconds_minimum_at_60kmh"
            },
            speed_requirements={
                "maximum_speed": "as_posted",
                "adjust_for_conditions": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "USA": "mph (miles per hour), varies by state",
                "Most_countries": "kmh (kilometers per hour)",
                "Germany_Autobahn": "Some sections have no speed limit",
                "UK": "mph, typically 30/60/70 limits"
            }
        ))
        
        # ============ HIGHWAY LANE DISCIPLINE ============
        self.add_rule(TrafficRule(
            rule_id="LANE_001",
            category="lane_management",
            trigger="multi_lane_highway_detected",
            description="Keep right/left except when overtaking, use passing lane appropriately",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.KEEP_RIGHT.value,  # or KEEP_LEFT depending on country
                ActionType.CHANGE_LANE.value
            ],
            conditions=[
                "slower_traffic_use_slow_lane",
                "overtake_in_passing_lane_only",
                "return_to_slow_lane_after_overtaking"
            ],
            exceptions=[
                "heavy_traffic_conditions",
                "preparing_for_exit"
            ],
            distance_requirements={
                "lane_change_signal": "100_meters_before_change",
                "safe_gap": "sufficient_space_both_lanes"
            },
            speed_requirements={
                "maintain_flow": True,
                "do_not_obstruct_passing_lane": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "USA_EU_UAE_China": "Keep right, pass left",
                "UK_Australia_India_Japan": "Keep left, pass right",
                "Germany": "Strict keep-right rule, passing on right is illegal"
            }
        ))
        
        # ============ NO OVERTAKING / NO PASSING ============
        self.add_rule(TrafficRule(
            rule_id="SIGN_003",
            category="traffic_signs",
            trigger="no_overtaking_sign_detected",
            description="Do not overtake/pass vehicles in this zone",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.INFORMATIONAL.value,
                ActionType.MAINTAIN_DISTANCE.value
            ],
            conditions=[
                "no_overtaking_zone",
                "dangerous_area_ahead",
                "limited_visibility"
            ],
            exceptions=[],
            distance_requirements={
                "maintain_lane": True,
                "do_not_cross_center_line": True
            },
            speed_requirements={},
            applicable_regions=["Global"],
            regional_variations={
                "UK_EU": "Red circular sign with two cars",
                "USA": "Yellow center line or 'Do Not Pass' sign",
                "Common_reasons": "Curves, hills, intersections, bridges"
            }
        ))
        
        # ============ ONE WAY STREET ============
        self.add_rule(TrafficRule(
            rule_id="SIGN_004",
            category="traffic_signs",
            trigger="one_way_sign_detected",
            description="Traffic flows in one direction only on this street",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.INFORMATIONAL.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "one_way_traffic_only",
                "do_not_enter_from_wrong_direction"
            ],
            exceptions=[],
            distance_requirements={},
            speed_requirements={},
            applicable_regions=["Global"],
            regional_variations={
                "Universal": "Arrow indicating direction of travel"
            }
        ))
        
        # ============ DO NOT ENTER / NO ENTRY ============
        self.add_rule(TrafficRule(
            rule_id="SIGN_005",
            category="traffic_signs",
            trigger="do_not_enter_sign_detected",
            description="Do not enter this road or area",
            priority=Priority.CRITICAL.value,
            actions=[
                ActionType.NO_ENTRY.value,
                ActionType.STOP.value
            ],
            conditions=[
                "wrong_way_entry",
                "one_way_street_from_wrong_direction",
                "restricted_area"
            ],
            exceptions=[],
            distance_requirements={},
            speed_requirements={
                "do_not_proceed": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "USA": "White rectangle with red circle 'Do Not Enter'",
                "EU": "White circle with red border (no entry)",
                "UK": "White circle with red border"
            }
        ))
        
        # ============ PRIORITY ROAD ============
        self.add_rule(TrafficRule(
            rule_id="SIGN_006",
            category="traffic_signs",
            trigger="priority_road_sign_detected",
            description="You are on a priority road, side traffic must yield to you",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.INFORMATIONAL.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "you_have_right_of_way",
                "side_roads_must_yield",
                "still_watch_for_violations"
            ],
            exceptions=[
                "traffic_signals_override",
                "emergency_vehicles"
            ],
            distance_requirements={},
            speed_requirements={
                "maintain_appropriate_speed": True
            },
            applicable_regions=["EU", "UK", "Global"],
            regional_variations={
                "EU": "Yellow diamond sign",
                "Less_common": "USA and some other countries"
            }
        ))
        
        # ============ CYCLISTS DETECTED ============
        self.add_rule(TrafficRule(
            rule_id="CYC_001",
            category="vulnerable_road_users",
            trigger="cyclist_detected",
            description="Give cyclists safe space when passing, reduce speed",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.MAINTAIN_DISTANCE.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "cyclist_on_road_or_bike_lane",
                "safe_passing_distance_required",
                "expect_cyclist_movement"
            ],
            exceptions=[],
            distance_requirements={
                "passing_clearance": "1_to_2_meters",
                "following_distance": "safe_distance_if_behind"
            },
            speed_requirements={
                "reduce_speed_when_passing": True,
                "be_prepared_to_stop": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "Netherlands": "Strong cyclist priority, extensive bike infrastructure",
                "USA": "3 feet minimum passing distance in most states",
                "UK": "1.5 meters minimum when passing",
                "Germany": "1.5-2 meters depending on speed"
            }
        ))
        
        # ============ HEAVY VEHICLES ============
        self.add_rule(TrafficRule(
            rule_id="HV_001",
            category="heavy_vehicles",
            trigger="heavy_vehicle_detected",
            description="Maintain increased following distance from trucks and buses",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.MAINTAIN_DISTANCE.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "truck_or_bus_ahead",
                "longer_stopping_distance",
                "limited_visibility_behind_truck"
            ],
            exceptions=[],
            distance_requirements={
                "following_distance": "4_to_6_seconds",
                "avoid_blind_spots": True
            },
            speed_requirements={},
            applicable_regions=["Global"],
            regional_variations={
                "USA": "If you can't see truck mirrors, driver can't see you",
                "EU": "Trucks limited to right lanes on highways"
            }
        ))
        
        # ============ FOG CONDITIONS ============
        self.add_rule(TrafficRule(
            rule_id="WEATHER_001",
            category="weather_conditions",
            trigger="fog_detected",
            description="Reduce speed significantly, use fog lights, increase following distance",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.MAINTAIN_DISTANCE.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "visibility_reduced",
                "fog_lights_on",
                "low_beam_headlights_on",
                "do_not_use_high_beams"
            ],
            exceptions=[
                "if_visibility_too_poor_pull_over"
            ],
            distance_requirements={
                "following_distance": "4_to_8_seconds",
                "stop_within_visible_range": True
            },
            speed_requirements={
                "reduce_speed_significantly": True,
                "adjust_to_visibility": "drive_at_speed_you_can_stop_within_sight"
            },
            applicable_regions=["Global"],
            regional_variations={
                "UK": "Fog lights must be used when visibility below 100m",
                "USA": "Use low beams, not high beams"
            }
        ))
        
        # ============ RAIN / WET ROADS ============
        self.add_rule(TrafficRule(
            rule_id="WEATHER_002",
            category="weather_conditions",
            trigger="rain_or_wet_road_detected",
            description="Reduce speed on wet roads, increase following distance, risk of hydroplaning",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.MAINTAIN_DISTANCE.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "wet_road_surface",
                "reduced_traction",
                "increased_braking_distance",
                "risk_of_hydroplaning"
            ],
            exceptions=[],
            distance_requirements={
                "following_distance": "4_to_6_seconds",
                "braking_distance": "doubled_on_wet_roads"
            },
            speed_requirements={
                "reduce_speed": "25_to_33_percent_reduction",
                "below_80_kmh_to_reduce_hydroplaning": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "Universal": "Wet roads double stopping distance"
            }
        ))
        
        # ============ SNOW / ICE CONDITIONS ============
        self.add_rule(TrafficRule(
            rule_id="WEATHER_003",
            category="weather_conditions",
            trigger="snow_or_ice_detected",
            description="Extreme caution, significantly reduced speed, very gentle inputs",
            priority=Priority.CRITICAL.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.MAINTAIN_DISTANCE.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "ice_or_snow_on_road",
                "severely_reduced_traction",
                "stopping_distance_multiplied",
                "gentle_steering_brake_acceleration"
            ],
            exceptions=[
                "consider_not_driving_if_conditions_severe"
            ],
            distance_requirements={
                "following_distance": "8_to_10_seconds",
                "stopping_distance": "10x_normal_on_ice"
            },
            speed_requirements={
                "reduce_speed": "50_to_75_percent_reduction",
                "drive_at_appropriate_speed_for_conditions": True
            },
            applicable_regions=["Northern_USA", "Canada", "Northern_EU", "Russia", "Japan", "Nordic_countries"],
            regional_variations={
                "Canada_Nordics": "Winter tires mandatory in some regions",
                "Germany_Austria": "Winter tire requirement Oct-Apr",
                "Japan": "Studded tires allowed in northern regions"
            }
        ))
        
        # ============ RAILROAD CROSSING ============
        self.add_rule(TrafficRule(
            rule_id="RAIL_001",
            category="rail_crossings",
            trigger="railroad_crossing_detected",
            description="Stop if signals active, never stop on tracks, watch for trains",
            priority=Priority.CRITICAL.value,
            actions=[
                ActionType.STOP.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "railroad_crossing_ahead",
                "stop_if_signals_active",
                "stop_if_gates_lowering",
                "stop_if_train_visible"
            ],
            exceptions=[
                "proceed_only_when_safe_and_clear"
            ],
            distance_requirements={
                "stop_distance": "15_to_50_feet_from_tracks",
                "never_stop_on_tracks": True,
                "ensure_full_vehicle_clears": True
            },
            speed_requirements={
                "approach_slowly": True,
                "be_prepared_to_stop": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "USA": "Stop 15-50 feet from nearest rail",
                "EU": "Red flashing lights mean stop",
                "India": "Many unmanned crossings, extreme caution required"
            }
        ))
        
        # ============ ANIMAL CROSSING ZONES ============
        self.add_rule(TrafficRule(
            rule_id="ANIMAL_001",
            category="special_zones",
            trigger="animal_crossing_sign_detected",
            description="Reduce speed, watch for animals crossing road",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "animal_crossing_zone",
                "wildlife_may_appear_suddenly",
                "increased_risk_dawn_dusk"
            ],
            exceptions=[],
            distance_requirements={
                "scan_road_sides": True
            },
            speed_requirements={
                "reduce_speed": True,
                "be_prepared_to_stop": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "USA_Canada": "Deer crossing common",
                "Australia": "Kangaroo crossing signs",
                "Nordic_countries": "Moose/elk crossings very dangerous",
                "UAE": "Camel crossing signs"
            }
        ))
        
        # ============ MOTORCYCLES DETECTED ============
        self.add_rule(TrafficRule(
            rule_id="MOTO_001",
            category="vulnerable_road_users",
            trigger="motorcycle_detected",
            description="Watch for motorcycles, check blind spots, give full lane width",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.MAINTAIN_DISTANCE.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "motorcycle_in_vicinity",
                "motorcycles_harder_to_see",
                "motorcycles_can_stop_faster",
                "vulnerable_in_collision"
            ],
            exceptions=[],
            distance_requirements={
                "following_distance": "3_to_4_seconds",
                "give_full_lane_width": True,
                "check_blind_spots_carefully": True
            },
            speed_requirements={
                "do_not_tailgate": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "SE_Asia": "Extremely high motorcycle density, constant awareness required",
                "USA_EU": "Lane splitting laws vary by region",
                "California": "Lane splitting legal for motorcycles"
            }
        ))
        
        # ============ CONSTRUCTION ZONE ============
        self.add_rule(TrafficRule(
            rule_id="ZONE_002",
            category="special_zones",
            trigger="construction_zone_sign_detected",
            description="Reduce speed, follow posted signs, watch for workers and equipment",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.PROCEED_WITH_CAUTION.value,
                ActionType.CHANGE_LANE.value
            ],
            conditions=[
                "construction_zone_ahead",
                "workers_present",
                "lane_shifts_possible",
                "reduced_speed_limit"
            ],
            exceptions=[],
            distance_requirements={
                "merge_early_if_lane_closed": True
            },
            speed_requirements={
                "follow_reduced_speed_limit": True,
                "doubled_fines_in_some_regions": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "USA": "Fines doubled in active work zones",
                "EU": "Strict reduced speed limits",
                "Many_countries": "Heavy fines for speeding in construction zones"
            }
        ))
        
        # ============ TUNNEL AHEAD ============
        self.add_rule(TrafficRule(
            rule_id="TUNNEL_001",
            category="special_zones",
            trigger="tunnel_sign_detected",
            description="Turn on headlights, maintain speed and lane, increase following distance",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.INFORMATIONAL.value,
                ActionType.MAINTAIN_DISTANCE.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            # CONTINUATION FROM TUNNEL_001

            conditions=[
                "tunnel_ahead",
                "turn_on_headlights",
                "maintain_speed_and_lane",
                "no_stopping_except_emergency"
            ],
            exceptions=[
                "emergency_situations_only"
            ],
            distance_requirements={
                "following_distance": "4_seconds_minimum",
                "do_not_change_lanes_unnecessarily": True
            },
            speed_requirements={
                "maintain_steady_speed": True,
                "follow_posted_tunnel_speed": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "EU": "Headlights mandatory in tunnels",
                "Switzerland_Austria": "Strict tunnel regulations",
                "Japan": "Many long tunnels, special rules apply"
            }
        ))
        
        # ============ BUS STOP / BUS LANE ============
        self.add_rule(TrafficRule(
            rule_id="BUS_001",
            category="special_lanes",
            trigger="bus_lane_detected",
            description="Do not drive in bus lanes during operational hours",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.INFORMATIONAL.value,
                ActionType.KEEP_RIGHT.value
            ],
            conditions=[
                "bus_lane_marking_present",
                "operational_hours_active",
                "buses_and_authorized_vehicles_only"
            ],
            exceptions=[
                "outside_operational_hours",
                "turning_at_intersection",
                "emergency_vehicles"
            ],
            distance_requirements={},
            speed_requirements={},
            applicable_regions=["Global"],
            regional_variations={
                "UK_London": "Red road surface, cameras enforce",
                "USA": "Often peak hours only",
                "Some_cities": "Taxis also allowed in bus lanes"
            }
        ))
        
        # ============ SCHOOL BUS WITH FLASHING LIGHTS ============
        self.add_rule(TrafficRule(
            rule_id="BUS_002",
            category="special_vehicles",
            trigger="school_bus_flashing_lights_detected",
            description="Stop when school bus has flashing red lights and stop arm extended",
            priority=Priority.CRITICAL.value,
            actions=[
                ActionType.STOP.value,
                ActionType.GIVE_WAY.value
            ],
            conditions=[
                "school_bus_loading_unloading",
                "red_lights_flashing",
                "stop_arm_extended",
                "children_crossing"
            ],
            exceptions=[
                "opposite_direction_on_divided_highway_with_barrier",
                "when_lights_turn_off_and_arm_retracts"
            ],
            distance_requirements={
                "stop_distance": "20_to_25_feet_from_bus",
                "remain_stopped_until_safe": True
            },
            speed_requirements={
                "must_remain_stopped": True
            },
            applicable_regions=["USA", "Canada"],
            regional_variations={
                "USA": "Both directions must stop except on divided highways",
                "Canada": "Similar to USA",
                "Most_other_countries": "No such specific rule"
            }
        ))
        
        # ============ MINIMUM SPEED LIMIT ============
        self.add_rule(TrafficRule(
            rule_id="SPEED_002",
            category="speed_management",
            trigger="minimum_speed_sign_detected",
            description="Must maintain at least the minimum posted speed unless conditions unsafe",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.INFORMATIONAL.value
            ],
            conditions=[
                "minimum_speed_posted",
                "typically_on_highways",
                "maintain_traffic_flow"
            ],
            exceptions=[
                "unsafe_weather_conditions",
                "traffic_congestion",
                "vehicle_malfunction"
            ],
            distance_requirements={},
            speed_requirements={
                "minimum_speed": "as_posted",
                "common_values": "40_to_60_kmh_on_highways"
            },
            applicable_regions=["USA", "EU", "Global"],
            regional_variations={
                "Germany_Autobahn": "60 kmh minimum on many sections",
                "USA": "Varies by state, typically on highways"
            }
        ))
        
        # ============ PARKING RESTRICTIONS ============
        self.add_rule(TrafficRule(
            rule_id="PARK_001",
            category="parking",
            trigger="no_parking_sign_detected",
            description="No parking allowed in this zone",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.INFORMATIONAL.value
            ],
            conditions=[
                "no_parking_zone",
                "stopping_may_be_allowed",
                "check_for_time_restrictions"
            ],
            exceptions=[
                "emergency_situations"
            ],
            distance_requirements={},
            speed_requirements={},
            applicable_regions=["Global"],
            regional_variations={
                "EU": "Blue circle with red border and red diagonal",
                "USA": "Various sign designs",
                "Japan": "Specific parking zones marked"
            }
        ))
        
        # ============ NO STOPPING ============
        self.add_rule(TrafficRule(
            rule_id="PARK_002",
            category="parking",
            trigger="no_stopping_sign_detected",
            description="No stopping or parking allowed at any time",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.INFORMATIONAL.value
            ],
            conditions=[
                "no_stopping_zone",
                "no_parking_allowed",
                "cannot_stop_even_briefly"
            ],
            exceptions=[
                "emergency_situations",
                "breakdown"
            ],
            distance_requirements={},
            speed_requirements={},
            applicable_regions=["Global"],
            regional_variations={
                "UK": "Red cross or red circle",
                "EU": "Blue circle with red border and cross",
                "Clearways": "Special no-stopping roads during peak hours"
            }
        ))
        
        # ============ CURVE AHEAD WARNING ============
        self.add_rule(TrafficRule(
            rule_id="WARN_001",
            category="warning_signs",
            trigger="curve_ahead_sign_detected",
            description="Sharp curve ahead, reduce speed before entering curve",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "sharp_curve_ahead",
                "reduce_speed_before_curve",
                "stay_in_lane"
            ],
            exceptions=[],
            distance_requirements={
                "slow_down_before_curve": "not_in_curve"
            },
            speed_requirements={
                "advisory_speed": "often_posted",
                "reduce_speed_appropriately": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "Universal": "Yellow warning signs with curve symbol"
            }
        ))
        
        # ============ STEEP GRADE / HILL ============
        self.add_rule(TrafficRule(
            rule_id="WARN_002",
            category="warning_signs",
            trigger="steep_grade_sign_detected",
            description="Steep hill ahead, use lower gear, watch for heavy vehicles",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "steep_grade_ahead",
                "use_lower_gear_for_control",
                "trucks_may_be_slow",
                "downhill_brake_overheating_risk"
            ],
            exceptions=[],
            distance_requirements={
                "watch_for_runaway_truck_ramps": True
            },
            speed_requirements={
                "control_speed_with_gears": True,
                "do_not_ride_brakes_downhill": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "Mountainous_regions": "Truck escape ramps common",
                "Percentage_or_ratio": "Grade steepness indicated"
            }
        ))
        
        # ============ SLIPPERY ROAD ============
        self.add_rule(TrafficRule(
            rule_id="WARN_003",
            category="warning_signs",
            trigger="slippery_road_sign_detected",
            description="Road may be slippery when wet, reduce speed and increase following distance",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.MAINTAIN_DISTANCE.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "slippery_conditions_possible",
                "wet_or_icy_surface",
                "reduced_traction"
            ],
            exceptions=[],
            distance_requirements={
                "following_distance": "increased"
            },
            speed_requirements={
                "reduce_speed": True,
                "gentle_inputs": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "Universal": "Car skidding symbol"
            }
        ))
        
        # ============ TRAFFIC MERGE ============
        self.add_rule(TrafficRule(
            rule_id="MERGE_001",
            category="lane_management",
            trigger="merge_sign_detected",
            description="Traffic merging ahead, adjust speed and position to allow merging",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.PROCEED_WITH_CAUTION.value,
                ActionType.MAINTAIN_DISTANCE.value,
                ActionType.CHANGE_LANE.value
            ],
            conditions=[
                "merging_traffic_ahead",
                "be_courteous_allow_merging",
                "zipper_merge_in_congestion"
            ],
            exceptions=[],
            distance_requirements={
                "leave_space_for_merging_vehicles": True
            },
            speed_requirements={
                "adjust_speed_to_facilitate_merge": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "Zipper_merge": "Take turns merging in congestion",
                "Some_regions": "Merging traffic must yield"
            }
        ))
        
        # ============ TRAFFIC CALMING - SPEED BUMPS ============
        self.add_rule(TrafficRule(
            rule_id="CALM_001",
            category="traffic_calming",
            trigger="speed_bump_detected",
            description="Slow down significantly for speed bump or speed hump",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.SLOW_DOWN.value
            ],
            conditions=[
                "speed_bump_ahead",
                "reduce_speed_to_avoid_damage",
                "common_in_residential_areas"
            ],
            exceptions=[],
            distance_requirements={},
            speed_requirements={
                "reduce_to": "5_to_20_kmh",
                "very_slow_speed_required": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "UK": "Speed humps common",
                "USA": "Speed bumps in parking lots and residential",
                "Some_countries": "Warning signs before bumps"
            }
        ))
        
        # ============ NARROW ROAD / BRIDGE ============
        self.add_rule(TrafficRule(
            rule_id="WARN_004",
            category="warning_signs",
            trigger="narrow_road_sign_detected",
            description="Road narrows ahead, reduce speed and prepare for reduced width",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.SLOW_DOWN.value,
                ActionType.PROCEED_WITH_CAUTION.value
            ],
            conditions=[
                "road_narrows",
                "may_need_to_yield",
                "oncoming_traffic_closer"
            ],
            exceptions=[],
            distance_requirements={
                "reduce_speed_before_narrow_section": True
            },
            speed_requirements={
                "reduce_speed": True,
                "be_prepared_to_stop": True
            },
            applicable_regions=["Global"],
            regional_variations={
                "Single_lane_bridges": "First to arrive has priority in some regions",
                "Priority_signs": "May indicate who yields"
            }
        ))
        
        # ============ TWO-WAY TRAFFIC AHEAD ============
        self.add_rule(TrafficRule(
            rule_id="WARN_005",
            category="warning_signs",
            trigger="two_way_traffic_sign_detected",
            description="Divided highway ends, two-way traffic begins",
            priority=Priority.MEDIUM.value,
            actions=[
                ActionType.PROCEED_WITH_CAUTION.value,
                ActionType.KEEP_RIGHT.value
            ],
            conditions=[
                "two_way_traffic_ahead",
                "oncoming_traffic_begins",
                "stay_in_proper_lane"
            ],
            exceptions=[],
            distance_requirements={
                "keep_right": True
            },
            speed_requirements={},
            applicable_regions=["Global"],
            regional_variations={
                "USA_EU": "Yellow diamond with two-way arrows",
                "Common_on_highways": "When divided highway ends"
            }
        ))
        
        # ============ PRIORITY TO ONCOMING TRAFFIC ============
        self.add_rule(TrafficRule(
            rule_id="PRIORITY_001",
            category="priority_rules",
            trigger="priority_to_oncoming_sign_detected",
            description="Give priority to oncoming traffic on narrow sections",
            priority=Priority.HIGH.value,
            actions=[
                ActionType.YIELD.value,
                ActionType.SLOW_DOWN.value
            ],
            conditions=[
                "narrow_section_ahead",
                "oncoming_traffic_has_priority",
                "wait_for_clear_path"
            ],
            exceptions=[
                "oncoming_traffic_yields_to_you"
            ],
            distance_requirements={
                "be_prepared_to_stop": True
            },
            speed_requirements={
                "reduce_speed": True
            },
            applicable_regions=["EU", "UK", "Global"],
            regional_variations={
                "Red_arrow": "You must give way",
                "Blue_arrow": "You have priority",
                "Common_on": "Narrow bridges and mountain roads"
            }
        ))

    def add_rule(self, rule: TrafficRule):
        """Add a traffic rule to the knowledge base"""
        self.rules.append(asdict(rule))
    
    def get_rules_by_category(self, category: str) -> List[Dict]:
        """Get all rules for a specific category"""
        return [rule for rule in self.rules if rule['category'] == category]
    
    def get_rule_by_trigger(self, trigger: str) -> Dict:
        """Get a specific rule by its trigger"""
        for rule in self.rules:
            if rule['trigger'] == trigger:
                return rule
        return None
    
    def get_rules_by_region(self, region: str) -> List[Dict]:
        """Get all rules applicable to a specific region"""
        return [rule for rule in self.rules if region in rule['applicable_regions'] or 'Global' in rule['applicable_regions']]
    
    def get_critical_rules(self) -> List[Dict]:
        """Get all critical priority rules"""
        return [rule for rule in self.rules if rule['priority'] == Priority.CRITICAL.value]
    
    def export_to_json(self, filename: str = "traffic_rules_knowledge_base.json"):
        """Export the knowledge base to JSON file"""
        output = {
            "version": "2.0",
            "description": "International Traffic Rules Knowledge Base",
            "total_rules": len(self.rules),
            "categories": list(set([rule['category'] for rule in self.rules])),
            "covered_regions": list(set([region for rule in self.rules for region in rule['applicable_regions']])),
            "rules": self.rules
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Knowledge base exported to {filename}")
        return output
    
    def generate_model_input_format(self) -> Dict:
        """Generate a format suitable for ML model input"""
        model_input = {
            "rule_mapping": {},
            "priority_levels": {
                "critical": [],
                "high": [],
                "medium": [],
                "low": []
            },
            "action_types": [action.value for action in ActionType],
            "detection_triggers": [],
            "category_index": {}
        }
        
        for rule in self.rules:
            trigger = rule['trigger']
            model_input["rule_mapping"][trigger] = {
                "rule_id": rule['rule_id'],
                "priority": rule['priority'],
                "actions": rule['actions'],
                "description": rule['description'],
                "conditions": rule['conditions'],
                "speed_requirements": rule['speed_requirements'],
                "distance_requirements": rule['distance_requirements']
            }
            
            model_input["priority_levels"][rule['priority']].append(rule['rule_id'])
            model_input["detection_triggers"].append(trigger)
            
            # Index by category
            category = rule['category']
            if category not in model_input["category_index"]:
                model_input["category_index"][category] = []
            model_input["category_index"][category].append(rule['rule_id'])
        
        return model_input
    
    def print_summary(self):
        """Print a summary of the knowledge base"""
        print("\n" + "="*70)
        print(" 🌍 INTERNATIONAL TRAFFIC RULES KNOWLEDGE BASE")
        print("="*70)
        print(f"📊 Total Rules: {len(self.rules)}")
        
        print(f"\n📂 Rules by Category:")
        categories = {}
        for rule in self.rules:
            cat = rule['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items()):
            print(f"   • {cat.replace('_', ' ').title()}: {count}")
        
        print(f"\n⚠️  Rules by Priority:")
        priorities = {}
        for rule in self.rules:
            pri = rule['priority']
            priorities[pri] = priorities.get(pri, 0) + 1
        
        for pri, count in sorted(priorities.items(), reverse=True):
            icon = "🔴" if pri == "critical" else "🟡" if pri == "high" else "🟢"
            print(f"   {icon} {pri.upper()}: {count}")
        
        print(f"\n🌐 Regions Covered:")
        regions = set()
        for rule in self.rules:
            regions.update(rule['applicable_regions'])
        print(f"   {', '.join(sorted(regions))}")
        
        print("="*70 + "\n")
    
    def print_regional_comparison(self, trigger: str):
        """Print how a rule varies across regions"""
        rule = self.get_rule_by_trigger(trigger)
        if not rule:
            print(f"Rule for trigger '{trigger}' not found")
            return
        
        print(f"\n{'='*70}")
        print(f"Regional Variations: {rule['description']}")
        print(f"{'='*70}")
        if rule['regional_variations']:
            for region, variation in rule['regional_variations'].items():
                print(f"\n{region.replace('_', ' ')}:")
                print(f"  → {variation}")
        else:
            print("\n  Universal rule - no significant regional variations")
        print(f"{'='*70}\n")


# Main execution
if __name__ == "__main__":
    # Create knowledge base
    print("\n🚗 Initializing International Traffic Rules Knowledge Base...")
    kb = TrafficRulesKnowledgeBase()
    
    # Print summary
    kb.print_summary()
    
    # Export to JSON
    print("📝 Exporting knowledge base...")
    kb_data = kb.export_to_json()
    
    # Generate model input format
    print("🤖 Generating ML model input format...")
    model_input = kb.generate_model_input_format()
    
    with open("traffic_rules_model_input.json", 'w', encoding='utf-8') as f:
        json.dump(model_input, f, indent=2, ensure_ascii=False)
    
    print("✓ Model input format exported to traffic_rules_model_input.json\n")
    
    # Example queries
    print("="*70)
    print("EXAMPLE QUERIES")
    print("="*70)
    
    # Example 1: Roundabout rule
    print("\n1️⃣  Example: Roundabout Rule")
    print("-" * 70)
    roundabout_rule = kb.get_rule_by_trigger("roundabout_sign_detected")
    print(json.dumps({
        "rule_id": roundabout_rule['rule_id'],
        "description": roundabout_rule['description'],
        "priority": roundabout_rule['priority'],
        "actions": roundabout_rule['actions'],
        "applicable_regions": roundabout_rule['applicable_regions']
    }, indent=2))
    
    # Regional variations for roundabouts
    kb.print_regional_comparison("roundabout_sign_detected")
    
    # Example 2: Emergency vehicle
    print("\n2️⃣  Example: Emergency Vehicle Rule")
    print("-" * 70)
    emergency_rule = kb.get_rule_by_trigger("emergency_vehicle_detected")
    print(json.dumps({
        "rule_id": emergency_rule['rule_id'],
        "description": emergency_rule['description'],
        "priority": emergency_rule['priority'],
        "actions": emergency_rule['actions']
    }, indent=2))
    
    # Example 3: All critical rules
    print("\n3️⃣  All Critical Priority Rules:")
    print("-" * 70)
    critical_rules = kb.get_critical_rules()
    for rule in critical_rules:
        print(f"   🔴 {rule['rule_id']}: {rule['description']}")
    
    print("\n" + "="*70)
    print("✅ Knowledge base generation complete!")
    print("="*70 + "\n")