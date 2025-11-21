(define (domain multi_agent_lines)
    (:requirements :fluents :adl :typing)
    (:types agent tank well - object)
    (:predicates (has-water ?a - agent) (has-filled ?a - agent))
    (:functions (xloc ?o - object) (yloc ?o - object) - integer
                (walls) - bit-matrix)

    (:action collect
     :parameters (?a - agent ?w - well)
     :precondition (and (not (has-water ?a))
                        (or (and (= (xloc ?a) (xloc ?w)) (= (- (yloc ?a) 1) (yloc ?w)))
                            (and (= (xloc ?a) (xloc ?w)) (= (+ (yloc ?a) 1) (yloc ?w)))
                            (and (= (- (xloc ?a) 1) (xloc ?w)) (= (yloc ?a) (yloc ?w)))
                            (and (= (+ (xloc ?a) 1) (xloc ?w)) (= (yloc ?a) (yloc ?w)))))
     :effect (has-water ?a) 
    )

    (:action fill
     :parameters (?a - agent ?t - tank)
     :precondition (and (has-water ?a)
                        (or (and (= (xloc ?a) (xloc ?t)) (= (- (yloc ?a) 1) (yloc ?t)))
                            (and (= (xloc ?a) (xloc ?t)) (= (+ (yloc ?a) 1) (yloc ?t)))
                            (and (= (- (xloc ?a) 1) (xloc ?t)) (= (yloc ?a) (yloc ?t)))
                            (and (= (+ (xloc ?a) 1) (xloc ?t)) (= (yloc ?a) (yloc ?t)))))
     :effect (and (not (has-water ?a)) (has-filled ?a))
    )

    (:action up
     :parameters (?a - agent)
     :precondition
        (and (> ypos 1)
            (= (get-index walls (- ypos 1) xpos) false)
            (not (exists (?t - tank)
                (and (= xpos (xloc ?t)) (= (- ypos 1) (yloc ?t)))))
            (not (exists (?w - well)
                (and (= xpos (xloc ?w)) (= (- ypos 1) (yloc ?w))))))
     :effect (decrease ypos 1)
    )
    (:action down
     :precondition
        (and (< ypos (height walls))
            (= (get-index walls (+ ypos 1) xpos) false)
            (not (exists (?t - tank)
                (and (= xpos (xloc ?t)) (= (+ ypos 1) (yloc ?t)))))
            (not (exists (?w - well)
                (and (= xpos (xloc ?w)) (= (+ ypos 1) (yloc ?w))))))
     :effect (increase ypos 1)
    )
    (:action left
     :precondition
        (and (> xpos 1)
            (= (get-index walls ypos (- xpos 1)) false)
            (not (exists (?t - tank)
                (and (= (- xpos 1) (xloc ?t)) (= ypos (yloc ?t)))))
            (not (exists (?w - well)
                (and (= (- xpos 1) (xloc ?w)) (= ypos (yloc ?w))))))
     :effect (decrease xpos 1)
    )
    (:action right
     :precondition
        (and (< xpos (width walls))
            (= (get-index walls ypos (+ xpos 1)) false)
            (not (exists (?t - tank)
                (and (= (+ xpos 1) (xloc ?t)) (= ypos (yloc ?t)))))
            (not (exists (?w - well)
                (and (= (+ xpos 1) (xloc ?w)) (= ypos (yloc ?w))))))
     :effect (increase xpos 1)
    )
)