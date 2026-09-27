import unittest
from tests import test_audit_layout as fixtures
from aaf_io.sdk_lane_layout import LaneLayoutEvent

class SdkAlignedOrderTests(unittest.TestCase):
    def run_group(self, classes, targets):
        lanes = [[fixtures.component('Filler',100), fixtures.component('SourceClip',20,str(i))] for i in range(3)]
        events = [LaneLayoutEvent(src_lane=i,top_idx=1,T_edit=100,L=20,target_lane=targets[i],visible_t=100,class_kind=classes[i],source_start=50,source_length=20) for i in range(3)]
        output = fixtures.LaneLayoutAuditTests().run_xml(fixtures.xml_document(lanes),events,n_lanes=3)
        return [node.findtext(fixtures.q('Payload')) for node in output.iter(fixtures.q('SourceClip'))]

    def test_sdk_restores_same_class_source_order(self):
        self.assertEqual(self.run_group(['speech']*3,[1,2,0]),['0','1','2'])

    def test_sdk_order_is_based_on_actual_class_not_physical_band(self):
        self.assertEqual(self.run_group(['speech','noise','speech'],[2,0,1]),['1','0','2'])
